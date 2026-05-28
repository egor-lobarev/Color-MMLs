import os
import math
import torch
import transformers.modeling_utils as _modeling_utils
# from transformers import Gemma4ForConditionalGeneration

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Monkey-patch caching_allocator_warmup — it ignores device_map and tries to
# pre-allocate the entire model on a single GPU, causing OOM on large models.
# Safe to disable: it's a performance hint, not required for correctness.
# ---------------------------------------------------------------------------
def _noop_warmup(model, device_map, hf_quantizer=None):
    pass

_modeling_utils.caching_allocator_warmup = _noop_warmup

# ---------------------------------------------------------------------------

from transformers import AutoConfig, AutoProcessor
from PIL import Image
from typing import Dict, List, Optional

from utils.embeddings.device_utils import (
    default_dtype,
    model_input_device,
    resolve_device,
)


def _get_free_memory_gb(gpu_idx: int) -> float:
    free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
    return free_bytes / 1024 ** 3


def _split_gemma4_device_map(model_name: str, headroom_gb: float = 10.0) -> Dict[str, int]:
    """
    Build device map using actual config parameter counts to estimate
    per-layer memory, rather than guessing from total model size.
    """
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name)
    text_cfg = getattr(config, "text_config", config)
    num_layers = int(text_cfg.num_hidden_layers)

    # --- Estimate per-layer size from actual architecture dimensions ---
    hidden  = int(getattr(text_cfg, "hidden_size",       5120))
    interm  = int(getattr(text_cfg, "intermediate_size", hidden * 4))
    n_heads = int(getattr(text_cfg, "num_attention_heads", 32))
    n_kv    = int(getattr(text_cfg, "num_key_value_heads", n_heads))
    head_d  = hidden // n_heads

    # Attention: Q + K + V + O projections
    attn_params = hidden * hidden + hidden * (n_kv * head_d) * 2 + hidden * hidden
    # MLP: gate + up + down (Gemma uses gated MLP)
    mlp_params  = hidden * interm * 3
    # Layer norms + scalars: negligible but count them
    norm_params = hidden * 6
    params_per_layer = attn_params + mlp_params + norm_params

    bytes_per_layer = params_per_layer * 2          # bfloat16 = 2 bytes
    layer_gb        = bytes_per_layer / 1024 ** 3
    # Add 15% overhead for optimizer state buffers, activations during load
    layer_gb *= 1.15

    # Vision tower: estimate from vision config if available
    vis_cfg    = getattr(config, "vision_config", None)
    vis_hidden = int(getattr(vis_cfg, "hidden_size",       1152) if vis_cfg else 1152)
    vis_layers = int(getattr(vis_cfg, "num_hidden_layers", 27)   if vis_cfg else 27)
    vis_params = vis_layers * vis_hidden * vis_hidden * 4   # rough ViT estimate
    vis_gb     = vis_params * 2 / 1024 ** 3 * 1.15

    # embed_tokens + lm_head + norms
    vocab_size    = int(getattr(text_cfg, "vocab_size", 256000))
    embed_gb      = vocab_size * hidden * 2 * 2 / 1024 ** 3   # embed + lm_head, bfloat16
    overhead_gpu0 = vis_gb + embed_gb + 1.0                    # +1 GiB for projector + norms

    # Real free memory
    free_gb = {}
    for i in range(world_size):
        free_bytes, _ = torch.cuda.mem_get_info(i)
        free_gb[i] = max(0.0, free_bytes / 1024 ** 3 - headroom_gb)

    print(f"  Gemma4 layer count    : {num_layers}")
    print(f"  Estimated layer size  : {layer_gb:.2f} GiB  (hidden={hidden}, interm={interm})")
    print(f"  Vision overhead GPU 0 : {vis_gb:.2f} GiB ViT + {embed_gb:.2f} GiB embed = {overhead_gpu0:.2f} GiB")
    for i in range(world_size):
        raw_free = free_gb[i] + headroom_gb
        print(f"  GPU {i}: {raw_free:.1f} GiB free → budget {free_gb[i]:.1f} GiB")

    # Pin non-layer components to GPU 0
    device_map: Dict[str, int] = {}
    for key in (
        "model.vision_tower",
        "model.embed_vision",
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "model.language_model.lm_head",
    ):
        device_map[key] = 0

    free_gb[0] = max(0.0, free_gb[0] - overhead_gpu0)

    # Greedy layer assignment — always pick GPU with most remaining budget
    gpu_used = {i: 0.0 for i in range(world_size)}
    for layer_idx in range(num_layers):
        best_gpu = max(range(world_size), key=lambda g: free_gb[g] - gpu_used[g])
        remaining = free_gb[best_gpu] - gpu_used[best_gpu]
        if remaining < layer_gb:
            # Print a warning but proceed — better than crashing before load
            print(f"  WARNING: GPU {best_gpu} only has {remaining:.2f} GiB left "
                  f"for layer {layer_idx} (~{layer_gb:.2f} GiB needed). "
                  f"Consider increasing headroom_gb or using more GPUs.")
        device_map[f"model.language_model.layers.{layer_idx}"] = best_gpu
        gpu_used[best_gpu] += layer_gb

    counts = {i: 0 for i in range(world_size)}
    for k, v in device_map.items():
        if "layers." in k:
            counts[v] += 1
    for i in range(world_size):
        print(f"  GPU {i}: {counts[i]} layers × {layer_gb:.2f} GiB = "
              f"{counts[i]*layer_gb:.1f} GiB model  |  budget was {free_gb[i]+overhead_gpu0*(i==0):.1f} GiB")

    return device_map

class Gemma4EmbeddingExtractor:
    def __init__(
        self,
        model_name: str = "google/gemma-4-27b-it",
        device: str = None,
        quantize_4_bit: bool = False,
        quantize_8_bit: bool = False,
        torch_dtype=None,
        system_prompt: Optional[str] = None,
        use_flash_attention: bool = False,
        headroom_gb: float = 8.0,
    ):
        if quantize_4_bit or quantize_8_bit:
            raise NotImplementedError(
                "Gemma4EmbeddingExtractor does not support bitsandbytes quantization yet."
            )

        self.system_prompt = system_prompt
        self.device = resolve_device(device)
        self.torch_dtype = default_dtype(self.device, torch_dtype)

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1 and self.device != "cpu":
            print(f"Gemma4EmbeddingExtractor: splitting across {n_gpus} GPUs.")
            device_map = _split_gemma4_device_map(model_name, headroom_gb=headroom_gb)
        elif self.device == "cpu":
            print("Gemma4EmbeddingExtractor: loading on CPU.")
            device_map = "cpu"
        else:
            device_map = {"": 0}

        # eager avoids sdpa's extra allocations; FA2 is opt-in
        attn_impl = (
            "flash_attention_2"
            if (use_flash_attention and self.device != "cpu")
            else "eager"
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._input_device = model_input_device(self.model)

        self.captures: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
        self._backbone = self.model.model
        self._register_hooks()

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _flatten(t: torch.Tensor) -> torch.Tensor:
        return t.squeeze(0) if t.dim() == 3 else t

    def _last_language_layer(self) -> torch.nn.Module:
        lm = self._backbone.language_model
        for attr in ("layers", "model.layers"):
            try:
                obj = lm
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj[-1]
            except AttributeError:
                continue
        raise AttributeError(
            f"Cannot locate decoder layers. "
            f"language_model children: {[n for n, _ in lm.named_children()]}"
        )

    def _register_hooks(self):
        vision_tower = getattr(self._backbone, "vision_tower", None)
        embed_vision  = getattr(self._backbone, "embed_vision", None)

        if vision_tower is None or embed_vision is None:
            available = [n for n, _ in self._backbone.named_children()]
            raise AttributeError(
                f"Missing vision_tower or embed_vision. Backbone children: {available}"
            )

        def hook_vision(_m, _inp, out):
            self.captures["vision_tokens"] = self._flatten(self._as_tensor(out)).detach()

        def hook_projector(_m, _inp, out):
            self.captures["projected_tokens"] = self._flatten(self._as_tensor(out)).detach()

        def hook_last_layer(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            self.captures["lm_last_hidden"] = hidden.detach()

        self.hooks.append(vision_tower.register_forward_hook(hook_vision))
        self.hooks.append(embed_vision.register_forward_hook(hook_projector))
        self.hooks.append(self._last_language_layer().register_forward_hook(hook_last_layer))

    def _build_inputs(self, images: List[Image.Image], prompt: str) -> Dict:
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })
        messages.append({
            "role": "user",
            "content": (
                [{"type": "image", "url": None, "image": im} for im in images]
                + [{"type": "text", "text": prompt}]
            ),
        })

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {
            k: v.to(self._input_device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

    @torch.no_grad()
    def extract(
        self,
        images: List[Image.Image],
        prompt: str = "Describe the images.",
        max_new_tokens: int = 256,
    ) -> Dict:
        self.captures.clear()
        inputs = self._build_inputs(images, prompt)

        self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        result = dict(self.captures)
        for key, pool_key, dim in (
            ("vision_tokens",    "vision_pooled_mean",    0),
            ("projected_tokens", "projected_pooled_mean", 0),
            ("lm_last_hidden",   "lm_pooled_mean",        1),
        ):
            if key in result:
                result[pool_key] = result[key].mean(dim=dim, keepdim=True)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        input_len = inputs["input_ids"].shape[1]
        result["model_answer"] = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )[0]

        return result

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()