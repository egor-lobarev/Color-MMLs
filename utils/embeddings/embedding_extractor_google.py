from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from PIL import Image
import torch
from typing import Dict, List, Optional


class Gemma4EmbeddingExtractor:
    """
    Embedding extractor for google/gemma-4-31B-it (and other Gemma 4 VL models).

    Architecture (confirmed from HF docs):
    - model.vision_tower              → SigLIP vision encoder
    - model.multi_modal_projector     → vision -> LLM projector
    - model.language_model.layers[-1] → last LLM decoder layer

    Captures:
    - "vision_tokens"          [Nv, Cv]   pre-projector SigLIP output
    - "projected_tokens"       [Nv, D]    post-projector, LLM hidden size
    - "lm_last_hidden"         [B, T, D]  last decoder layer hidden states
    + pooled means for all three
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-31B-it",
        device: str = None,
        torch_dtype=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self.model.eval()

        self.captures: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # 1) Vision tower (SigLIP encoder output, pre-projector)
        def hook_vision(_m, _inp, out):
            # SiglipVisionModel returns BaseModelOutputWithPooling
            # last_hidden_state shape: [B, Nv, Cv]
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            # flatten batch dim since B=1 in our use case → [Nv, Cv]
            self.captures["vision_tokens"] = hidden.squeeze(0).detach()

        self.hooks.append(
            self.model.vision_tower.register_forward_hook(hook_vision)
        )

        # 2) Multimodal projector (vision → LLM hidden size)
        def hook_projector(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            # shape: [Nv, D] or [B, Nv, D] — normalize
            if hidden.dim() == 3:
                hidden = hidden.squeeze(0)
            self.captures["projected_tokens"] = hidden.detach()

        self.hooks.append(
            self.model.multi_modal_projector.register_forward_hook(hook_projector)
        )

        # 3) Last LLM decoder layer
        # Gemma4: model.language_model.layers[-1]
        last_layer = self.model.language_model.layers[-1]

        def hook_last_layer(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            self.captures["lm_last_hidden"] = hidden.detach()  # [B, T, D]

        self.hooks.append(last_layer.register_forward_hook(hook_last_layer))

    def _build_inputs(
        self, images: List[Image.Image], prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Build inputs using Gemma 4 chat template."""

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "url": None, "image": im} for im in images]
                    + [{"type": "text", "text": prompt}]
                ),
            }
        ]

        # Gemma 4 processor handles apply_chat_template + tokenization in one call
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

    @torch.no_grad()
    def extract(
        self, images: List[Image.Image], prompt: str = "Describe the image."
    ) -> Dict:
        self.captures.clear()

        inputs = self._build_inputs(images, prompt)

        _ = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        result = dict(self.captures)

        # Pooled variants
        if "vision_tokens" in result:
            result["vision_pooled_mean"] = result["vision_tokens"].mean(dim=0, keepdim=True)
        if "projected_tokens" in result:
            result["projected_pooled_mean"] = result["projected_tokens"].mean(dim=0, keepdim=True)
        if "lm_last_hidden" in result:
            result["lm_pooled_mean"] = result["lm_last_hidden"].mean(dim=1, keepdim=True)

        # Generate answer
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
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
        self.hooks = []