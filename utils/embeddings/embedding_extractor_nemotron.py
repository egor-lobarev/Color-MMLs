import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from typing import Dict, List, Optional

from utils.embeddings.device_utils import (
    default_dtype,
    model_input_device,
    resolve_device,
)


class NemotronVLEmbeddingExtractor:
    """
    Embedding extractor for nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1.

    Architecture:
      - Vision encoder : C-RADIOv2-H  (NVIDIA custom ViT)
      - Projector      : MLP connecting vision→LLM space
      - LLM backbone   : Llama-3.1-8B-Instruct  (hidden size D=4096)

    Captures per forward pass:
      "vision_tokens"          [N_vis, C_vis]   C-RADIO vision encoder output
      "projected_tokens"       [N_vis, D]        after vision→LLM projector
      "lm_last_hidden"         [B, T, D]         last Llama decoder layer

    Plus mean-pooled variants and last-token LM vector:
      "vision_pooled_mean"     [1, 1, C_vis]
      "projected_pooled_mean"  [1, 1, D]
      "lm_pooled_mean"         [1, 1, D]     mean over full sequence
      "lm_last_token"          [1, 1, D]     last token only (better for classification)

    And the model's text reply:
      "model_answer"           str
    """

    def __init__(
        self,
        model_name: str = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
        device: str = None,
        quantize_4_bit: bool = False,
        quantize_8_bit: bool = False,
        torch_dtype=None,
        system_prompt: Optional[str] = None,
    ):
        if quantize_4_bit or quantize_8_bit:
            raise NotImplementedError(
                "NemotronVLEmbeddingExtractor does not support bitsandbytes quantization yet."
            )

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.device = resolve_device(device)
        self.torch_dtype = default_dtype(self.device, torch_dtype)

        if self.device == "cpu":
            print("NemotronVLEmbeddingExtractor: loading on CPU.")

        # Nemotron uses three separate objects: model, tokenizer, image_processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            device=self.device,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        ).eval()

        self._input_device = model_input_device(self.model)

        self.captures: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """
        Nemotron VL component paths (C-RADIO + Llama-3.1):

        The model is loaded via AutoModel with trust_remote_code — inspect
        children at runtime to find vision encoder and projector.

        Typical layout (based on NVIDIA LLaVA-style architecture):
          model.vision_tower     or model.visual  — C-RADIO vision encoder
          model.mm_projector     or model.merger  — MLP projector
          model.language_model.model.layers[-1]   — last Llama decoder layer
        """

        def _as_tensor(out):
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                return out.hidden_states[-1]
            return out[0] if isinstance(out, tuple) else out

        # 1) Vision encoder — try known attribute names
        vision_tower = None
        for attr in ("vision_tower", "visual", "vision_model", "image_encoder"):
            vision_tower = getattr(self.model, attr, None)
            if vision_tower is not None:
                print(f"NemotronVL: found vision encoder at model.{attr} "
                      f"({type(vision_tower).__name__})")
                break

        if vision_tower is None:
            available = [n for n, _ in self.model.named_children()]
            raise AttributeError(
                f"No vision encoder found. Model children: {available}"
            )

        def hook_vision(_m, _inp, out):
            hidden = _as_tensor(out)
            # Flatten tiles: [N_tiles, N_patch, C] → [N_tiles*N_patch, C]
            if hidden.dim() == 3:
                hidden = hidden.reshape(-1, hidden.shape[-1])
            self.captures["vision_tokens"] = hidden.detach()

        self.hooks.append(vision_tower.register_forward_hook(hook_vision))

        # 2) Projector — try known attribute names
        projector = None
        for attr in ("mm_projector", "merger", "multi_modal_projector",
                     "multimodal_projector", "projector", "mm_proj"):
            projector = getattr(self.model, attr, None)
            if projector is not None:
                print(f"NemotronVL: found projector at model.{attr} "
                      f"({type(projector).__name__})")
                break

        if projector is None:
            # Fallback: search named modules for "projector" or "merger"
            import re
            for name, mod in self.model.named_modules():
                if re.search(r"(project(or|er)|merger|mm_proj)", name, re.IGNORECASE):
                    projector = mod
                    print(f"NemotronVL: found projector via search at '{name}' "
                          f"({type(mod).__name__})")
                    break

        if projector is not None:
            def hook_projector(_m, _inp, out):
                hidden = _as_tensor(out)
                if hidden.dim() == 3:
                    hidden = hidden.reshape(-1, hidden.shape[-1])
                self.captures["projected_tokens"] = hidden.detach()

            self.hooks.append(projector.register_forward_hook(hook_projector))
        else:
            print("NemotronVL WARNING: no projector found — projected_tokens will be missing.")

        # 3) Last Llama decoder layer
        last_layer = self._find_last_decoder_layer()

        def hook_last_layer(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            self.captures["lm_last_hidden"] = hidden.detach()

        self.hooks.append(last_layer.register_forward_hook(hook_last_layer))

    def _find_last_decoder_layer(self) -> torch.nn.Module:
        """Locate the last Llama decoder layer across known path variants."""
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            # Some checkpoints embed the LM directly
            lm = self.model

        for path in ("model.layers", "layers", "decoder.layers"):
            try:
                obj = lm
                for part in path.split("."):
                    obj = getattr(obj, part)
                print(f"NemotronVL: found last decoder layer at language_model.{path}[-1]")
                return obj[-1]
            except AttributeError:
                continue

        raise AttributeError(
            f"Cannot locate decoder layers. "
            f"language_model children: {[n for n, _ in lm.named_children()]}"
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_images(self, images: List[Image.Image]) -> Dict:
        """
        Use AutoImageProcessor to tile and tokenize images.
        Returns a dict with pixel_values and related keys, on the model device.
        """
        image_features = self.image_processor(images)
        return {
            k: v.to(self._input_device) if torch.is_tensor(v) else v
            for k, v in image_features.items()
        }

    def _build_question(self, n_images: int, prompt: str) -> str:
        """Prepend system prompt if set."""
        if self.system_prompt:
            return f"{self.system_prompt}\n{prompt}"
        return prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(
        self,
        images: List[Image.Image],
        prompt: str = "Describe the images.",
        max_new_tokens: int = 256,
    ) -> Dict:
        """
        Run a forward + generation pass and return embeddings + answer.

        Returns dict with:
          vision_tokens, projected_tokens, lm_last_hidden   (raw)
          vision_pooled_mean, projected_pooled_mean          [1, 1, C]
          lm_pooled_mean, lm_last_token                      [1, 1, D]
          model_answer                                        str
        """
        self.captures.clear()

        image_features = self.preprocess_images(images)
        question = self._build_question(len(images), prompt)

        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # model.chat handles prompt formatting + forward + generate internally;
        # hooks fire during the forward pass inside chat()
        response = self.model.chat(
            tokenizer=self.tokenizer,
            question=question,
            generation_config=generation_config,
            **image_features,
        )

        result = dict(self.captures)

        # Normalize all raw tensors to [1, N_tokens, C] for uniform pooling
        for key in ("vision_tokens", "projected_tokens"):
            if key in result:
                t = result[key]
                if t.dim() == 2:
                    t = t.unsqueeze(0)          # [N, C] → [1, N, C]
                result[key] = t

        if "lm_last_hidden" in result:
            t = result["lm_last_hidden"]
            if t.dim() == 2:
                t = t.unsqueeze(0)
            result["lm_last_hidden"] = t

        # Pooled means — always over dim=1 (token dimension)
        for key, pool_key in (
            ("vision_tokens",    "vision_pooled_mean"),
            ("projected_tokens", "projected_pooled_mean"),
            ("lm_last_hidden",   "lm_pooled_mean"),
        ):
            if key in result:
                result[pool_key] = result[key].mean(dim=1, keepdim=True)  # [1, 1, C]

        # Last token — best single-vector summary for decoder-only LLMs
        if "lm_last_hidden" in result:
            result["lm_last_token"] = result["lm_last_hidden"][:, -1:, :]  # [1, 1, D]

        result["model_answer"] = response
        return result

    def remove_hooks(self):
        """Release all registered forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()