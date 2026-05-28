import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple

from utils.embeddings.device_utils import (
    model_input_device,
    resolve_device,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_ar = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, image_size
    )
    target_w = image_size * target_ar[0]
    target_h = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]

    resized = image.resize((target_w, target_h))
    tiles = []
    cols = target_w // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


def preprocess_image(
    image: Image.Image,
    input_size: int = 448,
    max_num: int = 12,
) -> torch.Tensor:
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


def _split_model_device_map(model_name: str) -> Dict[str, int]:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    world_size = torch.cuda.device_count()

    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    counts = [num_layers_per_gpu] * world_size
    counts[0] = math.ceil(counts[0] * 0.5)

    device_map: Dict[str, int] = {}
    layer_cnt = 0
    for gpu_idx, n in enumerate(counts):
        for _ in range(n):
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_idx
            layer_cnt += 1

    for key in (
        "vision_model",
        "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.output",
        "language_model.model.norm",
        "language_model.model.rotary_emb",
        "language_model.lm_head",
        f"language_model.model.layers.{num_layers - 1}",
    ):
        device_map[key] = 0

    return device_map


class InternVL3EmbeddingExtractor:
    """
    Embedding extractor for OpenGVLab/InternVL3-* models.

    Captures per forward pass:
      "vision_tokens"          [1, N_vis, C]    vision encoder patch tokens (no CLS)
      "projected_tokens"       [1, N_vis, D]    after MLP1 projector
      "lm_last_hidden"         [1, T, D]        last decoder layer (prefill only)

    Pooled variants (all shape [1, 1, C/D]):
      "vision_pooled_mean"     mean over N_vis patch tokens
      "projected_pooled_mean"  mean over N_vis projected tokens
      "lm_pooled_mean"         masked mean over real tokens (excludes EOS/PAD)
      "lm_last_token"          last real token (better than mean for classification)
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B",
        device: str = None,
        quantize_8_bit: bool = False,
        quantize_4_bit: bool = False,
        use_flash_attn: bool = True,
        torch_dtype=None,
        input_size: int = 448,
        max_tiles: int = 12,
        system_prompt: Optional[str] = None,
    ):
        if quantize_4_bit:
            raise NotImplementedError("4-bit quantization is not supported yet.")

        self.model_name = model_name
        self.input_size = input_size
        self.max_tiles = max_tiles
        self.system_prompt = system_prompt
        self.device = resolve_device(device)
        self.torch_dtype = torch_dtype

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1 and self.device != "cpu":
            print(f"InternVL3EmbeddingExtractor: splitting model across {n_gpus} GPUs.")
            device_map = _split_model_device_map(model_name)
        else:
            device_map = self.device

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        self._input_device = model_input_device(self.model)

        self.captures: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _register_hooks(self):

        def _as_tensor(out):
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state
            return out[0] if isinstance(out, tuple) else out

        # 1) Vision encoder — drop CLS, normalize to [1, N_patches, C]
        vision_model = getattr(self.model, "vision_model", None)
        if vision_model is None:
            available = [n for n, _ in self.model.named_children()]
            raise AttributeError(f"No vision_model found. Children: {available}")

        def hook_vision(_m, _inp, out):
            hidden = _as_tensor(out)                          # [N_tiles, N_patch+1, C]
            patch = hidden[:, 1:, :] if hidden.dim() == 3 else hidden  # drop CLS
            # Flatten tiles into token sequence → [1, N_tiles*N_patch, C]
            flat = patch.reshape(1, -1, patch.shape[-1])
            self.captures["vision_tokens"] = flat.detach().cpu()

        self.hooks.append(vision_model.register_forward_hook(hook_vision))

        # 2) MLP1 projector — normalize to [1, N_projected, D]
        mlp1 = getattr(self.model, "mlp1", None)
        if mlp1 is None:
            available = [n for n, _ in self.model.named_children()]
            raise AttributeError(f"No mlp1 projector found. Children: {available}")

        def hook_mlp1(_m, _inp, out):
            hidden = _as_tensor(out)                          # [N_vis, D] or [1, N_vis, D]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)                  # → [1, N_vis, D]
            self.captures["projected_tokens"] = hidden.detach().cpu()

        self.hooks.append(mlp1.register_forward_hook(hook_mlp1))

        # 3) Last LLM decoder layer
        #    Also capture attention_mask from the layer's INPUT so we can do
        #    masked pooling later — avoids averaging over EOS/PAD tokens.
        try:
            last_layer = self.model.language_model.model.layers[-1]
        except AttributeError:
            last_layer = self.model.language_model.model.decoder.layers[-1]

        def hook_last_layer(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out  # [B, T, D]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            self.captures["lm_last_hidden"] = hidden.detach().cpu()

            # Grab attention_mask from the layer's keyword inputs if available.
            # InternVL/Qwen2 decoder layers receive it as a keyword arg.
            # _inp is a tuple of positional args; kwargs aren't directly exposed
            # via forward hooks, so we rely on the hidden state length instead
            # (see _masked_pool below).

        self.hooks.append(last_layer.register_forward_hook(hook_last_layer))

    # ------------------------------------------------------------------
    # Pooling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_pool(
        hidden: torch.Tensor,           # [1, T, D]  on CPU
        eos_token_id: int,
        input_ids: Optional[torch.Tensor],  # [1, T] token ids, CPU
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          lm_pooled_mean  [1, 1, D]  mean over real (non-EOS/PAD) tokens
          lm_last_token   [1, 1, D]  last real token
        """
        T = hidden.shape[1]

        if input_ids is not None and input_ids.shape[1] == T:
            # Build mask: 1 for real tokens, 0 for EOS/PAD
            # InternVL uses eos_token_id as pad_token_id
            mask = (input_ids != eos_token_id).float()   # [1, T]

            # Last real token index
            real_lengths = mask.sum(dim=1).long()        # [1]
            last_idx = (real_lengths - 1).clamp(min=0)  # [1]
        else:
            # Fallback: treat all tokens as real
            mask = torch.ones(1, T, dtype=torch.float32)
            last_idx = torch.tensor([T - 1])

        # Masked mean
        mask_3d = mask.unsqueeze(-1)                     # [1, T, 1]
        pooled_mean = (hidden * mask_3d).sum(dim=1, keepdim=True) / mask_3d.sum(dim=1, keepdim=True).clamp(min=1)

        # Last real token
        last_token = hidden[
            torch.arange(hidden.shape[0]), last_idx
        ].unsqueeze(1)                                   # [1, 1, D]

        return pooled_mean, last_token

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_images(
        self, images: List[Image.Image]
    ) -> Tuple[torch.Tensor, List[int]]:
        tiles_per_image = [
            preprocess_image(img, self.input_size, self.max_tiles)
            for img in images
        ]
        num_patches = [t.shape[0] for t in tiles_per_image]
        pixel_values = torch.cat(tiles_per_image, dim=0).to(
            dtype=self.torch_dtype, device=self._input_device
        )
        return pixel_values, num_patches

    def _build_question(self, n_images: int, prompt: str) -> str:
        if n_images == 1:
            return f"<image>\n{prompt}"
        tags = "".join(f"Image-{i + 1}: <image>\n" for i in range(n_images))
        return f"{tags}{prompt}"

    def _get_input_ids(self, question: str, pixel_values: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Tokenize the prefill prompt (without generation) so we have input_ids
        for masked pooling. Mirrors what model.chat() does internally.
        """
        try:
            IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
            img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            num_image_tokens = pixel_values.shape[0] * self.model.num_image_token

            # Build the prompt string the same way InternVL's chat() does
            if self.system_prompt:
                system = self.system_prompt
            else:
                system = self.model.system_message if hasattr(self.model, "system_message") else ""

            template = self.model.conv_template if hasattr(self.model, "conv_template") else None
            if template is not None:
                # Use model's own conversation template for exact match
                from copy import deepcopy
                conv = deepcopy(template)
                conv.system_message = system
                image_tokens = IMG_CONTEXT_TOKEN * num_image_tokens
                conv.append_message(conv.roles[0], image_tokens + "\n" + question)
                conv.append_message(conv.roles[1], None)
                query = conv.get_prompt()
            else:
                query = question

            input_ids = self.tokenizer(
                query, return_tensors="pt", add_special_tokens=False
            ).input_ids                                   # [1, T_prefill]
            return input_ids
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(
        self,
        images: List[Image.Image],
        prompt: str = "Describe the images.",
        generation_config: Optional[Dict] = None,
        history: Optional[List] = None,
    ) -> Dict:
        self.captures.clear()

        if generation_config is None:
            generation_config = {"max_new_tokens": 256, "do_sample": False}

        pixel_values, num_patches = self.preprocess_images(images)
        question = self._build_question(len(images), prompt)

        # Try to build input_ids for masked pooling before chat() runs
        input_ids = self._get_input_ids(question, pixel_values)

        response, new_history = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches if len(images) > 1 else None,
            history=history,
            return_history=True,
        )

        result = dict(self.captures)

        # --- Vision & projected: mean over token dim=1 → [1, 1, C] ---
        for key, pool_key in (
            ("vision_tokens",    "vision_pooled_mean"),
            ("projected_tokens", "projected_pooled_mean"),
        ):
            if key in result:
                t = result[key]                          # already [1, N, C]
                result[pool_key] = t.mean(dim=1, keepdim=True)   # [1, 1, C]

        # --- LM: masked mean + last real token → [1, 1, D] ---
        if "lm_last_hidden" in result:
            hidden = result["lm_last_hidden"]            # [1, T, D] on CPU
            eos_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("<|im_end|>")

            lm_mean, lm_last = self._masked_pool(hidden, eos_id, input_ids)
            result["lm_pooled_mean"] = lm_mean           # [1, 1, D]
            result["lm_last_token"]  = lm_last           # [1, 1, D]

        result["model_answer"] = response
        result["history"] = new_history
        return result

    @torch.no_grad()
    def extract_batch(
        self,
        batch_images: List[List[Image.Image]],
        prompts: List[str],
        generation_config: Optional[Dict] = None,
    ) -> List[Dict]:
        if generation_config is None:
            generation_config = {"max_new_tokens": 256, "do_sample": False}

        all_tiles, num_patches_list, questions = [], [], []

        for images, prompt in zip(batch_images, prompts):
            pv, np_ = self.preprocess_images(images)
            all_tiles.append(pv)
            num_patches_list.extend(np_)
            questions.append(self._build_question(len(images), prompt))

        pixel_values = torch.cat(all_tiles, dim=0)

        self.captures.clear()
        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )

        raw = dict(self.captures)
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        results = []
        for response in responses:
            r = dict(raw)
            for key, pool_key in (
                ("vision_tokens",    "vision_pooled_mean"),
                ("projected_tokens", "projected_pooled_mean"),
            ):
                if key in raw:
                    r[pool_key] = raw[key].mean(dim=1, keepdim=True)

            if "lm_last_hidden" in raw:
                lm_mean, lm_last = self._masked_pool(raw["lm_last_hidden"], eos_id, None)
                r["lm_pooled_mean"] = lm_mean
                r["lm_last_token"]  = lm_last

            r["model_answer"] = response
            results.append(r)

        return results

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()