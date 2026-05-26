from transformers import AutoProcessor, Phi4MultimodalForCausalLM
from PIL import Image
import torch
from typing import Dict, List

class Phi4EmbeddingExtractor:
    """
    Embedding extractor for microsoft/Phi-4-multimodal-instruct.

    Captures:
    - "vision_tokens"         [Nv, Cv]   vision encoder output (pre-projector)
    - "projected_tokens"      [Nv, D]    after vision->LLM projector
    - "lm_last_hidden"        [B, T, D]  last LLM decoder layer
    - pooled means for all three
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-multimodal-instruct",
        device: str = None,
        quantize_4_bit: bool = False,
        quantize_8_bit: bool = False,
        torch_dtype=None,
        system_prompt=None,
    ):
        if quantize_4_bit or quantize_8_bit:
            raise NotImplementedError(
                "Phi4EmbeddingExtractor does not support bitsandbytes quantization yet."
            )
        self.system_prompt = system_prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = Phi4MultimodalForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        # Load vision LoRA adapter — required for image inputs
        self.model.load_adapter(
            model_name,
            adapter_name="vision",
            adapter_kwargs={"subfolder": "vision-lora"}
        )
        self.model.set_adapter("vision")
        self.model.eval()

        self.captures: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # 1) Vision encoder output (pre-projector)
        # Phi-4 vision encoder is at model.model.vision_embed_tokens.img_processor
        vision_encoder = self.model.model.vision_embed_tokens.img_processor

        def hook_vision(_m, _inp, out):
            # out shape: [Nv, Cv]
            hidden = out[0] if isinstance(out, tuple) else out
            self.captures["vision_tokens"] = hidden.detach()

        self.hooks.append(vision_encoder.register_forward_hook(hook_vision))

        # 2) Vision projector (maps vision -> LLM hidden size)
        # In Phi-4 this is vision_embed_tokens itself (the full module including projection)
        projector = self.model.model.vision_embed_tokens

        def hook_projector(_m, _inp, out):
            # out shape: [Nv, D]
            hidden = out[0] if isinstance(out, tuple) else out
            self.captures["projected_tokens"] = hidden.detach()

        self.hooks.append(projector.register_forward_hook(hook_projector))

        # 3) Last LLM decoder layer
        last_layer = self.model.model.layers[-1]

        def hook_last_layer(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            self.captures["lm_last_hidden"] = hidden.detach()

        self.hooks.append(last_layer.register_forward_hook(hook_last_layer))

    def _build_inputs(self, images: List[Image.Image], prompt: str) -> Dict:
        """Build inputs using Phi-4's chat template format."""

        # Phi-4 uses <|image_N|> placeholders in the prompt
        image_tags = "".join([f"<|image_{i+1}|>" for i in range(len(images))])

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": f"{image_tags}\n{prompt}",
        })

        # apply_chat_template with tokenize=True returns inputs directly for Phi-4
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images=images,
        )

        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

    @torch.no_grad()
    def extract(
        self, images: List[Image.Image], prompt: str = "Describe the images."
    ) -> Dict[str, torch.Tensor]:
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
            skip_special_tokens=True
        )[0]

        return result

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []