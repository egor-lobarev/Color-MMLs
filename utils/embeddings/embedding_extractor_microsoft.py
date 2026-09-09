from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from PIL import Image
import torch
from typing import Any, Dict, List

from utils.embeddings.device_utils import (
    default_dtype,
    hf_device_map,
    model_input_device,
    resolve_device,
)


class Phi4EmbeddingExtractor:
    """
    Embedding extractor for microsoft/Phi-4-multimodal-instruct.

    Captures:
    - "vision_tokens"         [Nv, Cv]   vision encoder output (pre-projector)
    - "projected_tokens"      [Nv, D]    after vision->LLM projector
    - "lm_last_hidden"        [B, T, D]  last LLM decoder layer
    - pooled means for all three
    """
    # Сколько токенов генерировать после прямого прохода. Эмбеддинги
    # снимаются хуками на prefill, поэтому для их извлечения ответ модели не
    # нужен: скрипты выставляют 1, что убирает авторегрессионный декод
    # (~30-60 токенов на цвет) и ускоряет прогон примерно на порядок.
    max_new_tokens: int = 256

    USER_PROMPT = "<|user|>"
    ASSISTANT_PROMPT = "<|assistant|>"
    PROMPT_SUFFIX = "<|end|>"

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-multimodal-instruct",
        device: str = None,
        quantize_4_bit: bool = False,
        quantize_8_bit: bool = False,
        torch_dtype=None,
        system_prompt: str = None,
        use_flash_attention: bool = False,
    ):
        if quantize_4_bit or quantize_8_bit:
            raise NotImplementedError(
                "Phi4EmbeddingExtractor does not support bitsandbytes quantization yet."
            )

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.device = resolve_device(device)
        self.torch_dtype = default_dtype(self.device, torch_dtype)

        if self.device == "cpu":
            print(
                "Phi4EmbeddingExtractor: loading on CPU "
                "(CUDA unavailable or incompatible with this PyTorch build)."
            )

        # Processor — trust_remote_code required for Phi-4 multimodal
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Attention implementation: flash_attention_2 requires Ampere+ GPU
        attn_impl = "flash_attention_2" if (use_flash_attention and self.device != "cpu") else "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            _attn_implementation=attn_impl,
        )

        # Load vision LoRA adapter — required for image inputs in many Phi-4 checkpoints
        self._vision_lora_loaded = False
        try:
            self.model.load_adapter(
                model_name,
                adapter_name="vision",
                adapter_kwargs={"subfolder": "vision-lora"},
            )
            self._vision_lora_loaded = True
        except (ImportError, Exception):
            pass

        if self._vision_lora_loaded:
            try:
                self.model.set_adapter("vision")
            except Exception:
                pass

        self.model.eval()
        self._input_device = model_input_device(self.model)
        self.generation_config = GenerationConfig.from_pretrained(model_name)

        self.captures: Dict[str, torch.Tensor] = {}
        self.hooks: list = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Forward hooks
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """
        Hook into:
          - vision tower       → captures["vision_tokens"]
          - img_projection_down → captures["projected_tokens"]
          - last decoder layer → captures["lm_last_hidden"]
        """

        def _as_tensor(out):
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                return out.last_hidden_state
            if isinstance(out, tuple):
                return out[0]
            return out

        mm = self.model.model
        embed_ext = getattr(mm, "embed_tokens_extend", None)
        if embed_ext is None or not hasattr(embed_ext, "image_embed"):
            raise AttributeError(
                "Phi4MultimodalModel has no embed_tokens_extend.image_embed — unexpected architecture."
            )
        image_embed = embed_ext.image_embed

        # 1) Vision tower
        vision_tower = getattr(image_embed, "img_processor", None)
        if vision_tower is None:
            raise AttributeError("Phi4MultimodalImageEmbedding has no img_processor.")

        def hook_vision(_m, _inp, out):
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                idx = getattr(image_embed, "layer_idx", -1)
                hidden = out.hidden_states[idx]
            else:
                hidden = _as_tensor(out)
            self.captures["vision_tokens"] = (
                hidden.squeeze(0).detach() if hidden.dim() == 3 else hidden.detach()
            )

        self.hooks.append(vision_tower.register_forward_hook(hook_vision))

        # 2) Projection layer
        proj_down = getattr(image_embed, "img_projection", None)
        if proj_down is None:
            raise AttributeError("Phi4MultimodalImageEmbedding has no img_projection.")

        def hook_projector(_m, _inp, out):
            self.captures["projected_tokens"] = _as_tensor(out).detach()

        self.hooks.append(proj_down.register_forward_hook(hook_projector))

        # 3) Last LLM decoder layer
        def hook_last_layer(_m, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            self.captures["lm_last_hidden"] = hidden.detach()

        self.hooks.append(self.model.model.layers[-1].register_forward_hook(hook_last_layer))

    # ------------------------------------------------------------------
    # Prompt / input helpers  (aligned with HF guide pattern)
    # ------------------------------------------------------------------

    def _build_prompt(self, n_images: int, text: str) -> str:
        """
        Build the raw prompt string using Phi-4's special tokens, matching
        the HuggingFace guide's format exactly:

            <|user|><|image_1|>...<|image_N|>{text}<|end|><|assistant|>
        """
        image_tags = "".join(f"<|image_{i + 1}|>" for i in range(n_images))

        system_block = ""
        if self.system_prompt:
            system_block = f"<|system|>{self.system_prompt}{self.PROMPT_SUFFIX}"

        return (
            f"{system_block}"
            f"{self.USER_PROMPT}"
            f"{image_tags}"
            f"{text}"
            f"{self.PROMPT_SUFFIX}"
            f"{self.ASSISTANT_PROMPT}"
        )

    def _build_inputs(self, images: List[Image.Image], prompt_text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenise using the simple processor(text=..., images=...) call
        shown in the HuggingFace guide — no chat-template branching needed.
        """
        prompt = self._build_prompt(len(images), prompt_text)

        # Single image: pass the Image object directly (HF guide style).
        # Multiple images: pass as a list.
        image_input = images[0] if len(images) == 1 else images

        inputs = self.processor(
            text=prompt,
            images=image_input,
            return_tensors="pt",
        )

        return {
            k: v.to(self._input_device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(
        self,
        images: List[Image.Image],
        prompt: str = "Describe the images.",
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass (+ generation) and return raw + pooled embeddings
        together with the model's text response.
        """
        self.captures.clear()

        inputs = self._build_inputs(images, prompt)

        if self._vision_lora_loaded:
            self.model.enable_adapters()

        # Forward pass — captures embeddings via hooks
        self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        result = dict(self.captures)

        # Pooled means.
        # Усреднять нужно по ОСИ ТОКЕНОВ. Прежний код брал mean(dim=0) и на
        # реальных формах не пулил вовсе: vision_tokens [2, 1024, 1152] (2 тайла
        # HD-трансформа) усреднялись по тайлам -> [1, 1024, 1152], а
        # projected_tokens [1, 545, 3072] давали no-op -> [1, 545, 3072].
        # В результате «pooled»-файлы содержали сотни векторов вместо одного
        # (~11 МиБ на цвет) и требовали отдельного прогона
        # data/embeddings/compress_embeddings.py. Схлопываем все ведущие оси и
        # приводим к [1, 1, D] — той же форме, что даёт compress_embeddings.py.
        def _pool_tokens(t):
            return t.reshape(-1, t.shape[-1]).mean(dim=0).reshape(1, 1, -1)

        if "vision_tokens" in result:
            result["vision_pooled_mean"] = _pool_tokens(result["vision_tokens"])
        if "projected_tokens" in result:
            result["projected_pooled_mean"] = _pool_tokens(result["projected_tokens"])
        if "lm_last_hidden" in result:
            # здесь ось токенов — dim=1, пулинг уже был корректным
            result["lm_pooled_mean"] = result["lm_last_hidden"].mean(dim=1, keepdim=True)

        # Generation — keep adapter enabled (same session)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            generation_config=self.generation_config,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        if self._vision_lora_loaded:
            self.model.disable_adapters()

        input_len = inputs["input_ids"].shape[1]
        result["model_answer"] = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return result

    def remove_hooks(self):
        """Call this when done to free hook memory."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    # Единый API экстракторов: фасад EmbeddingsExtractor вызывает close() в
    # блоке finally, поэтому без этого псевдонима прогон падал бы в самом конце
    # (AttributeError), теряя _run_log.json и список ошибок.
    close = remove_hooks