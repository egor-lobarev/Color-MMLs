from typing import Dict, List, Type

import torch
from PIL import Image

from utils.embeddings.embedding_extractor_google import Gemma4EmbeddingExtractor
from utils.embeddings.embedding_extractor_microsoft import Phi4EmbeddingExtractor
from utils.embeddings.embedding_extractor_qwen import Qwen25VLEmbeddingExtractor
from utils.embeddings.embedding_extractor_internvl3 import InternVL3EmbeddingExtractor

__all__ = [
    "EmbeddingsExtractor",
    "Gemma4EmbeddingExtractor",
    "Phi4EmbeddingExtractor",
    "Qwen25VLEmbeddingExtractor",
    "resolve_extractor_class",
]


def resolve_extractor_class(model_name: str) -> Type:
    """
    Map a Hugging Face model id (or local path) to a concrete extractor class.

    Resolution order: model id heuristics, then ``AutoConfig.model_type`` /
    ``architectures``, then error.
    """
    name_lower = model_name.lower().replace("\\", "/")

    if "qwen" in name_lower and ("vl" in name_lower or "2.5" in name_lower or "2_5" in name_lower):
        return Qwen25VLEmbeddingExtractor
    if "gemma" in name_lower and "4" in name_lower:
        return Gemma4EmbeddingExtractor
    if "gemma" in name_lower:
        return Gemma4EmbeddingExtractor
    if "phi" in name_lower and "multimodal" in name_lower:
        return Phi4EmbeddingExtractor
    if "phi-4" in name_lower or "phi4" in name_lower:
        return Phi4EmbeddingExtractor
    if "internvl3" in name_lower:
        return InternVL3EmbeddingExtractor

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = (getattr(config, "model_type", None) or "").lower()
        arch_blob = " ".join(getattr(config, "architectures", None) or []).lower()

        if "qwen2_5_vl" in model_type or "qwen2_5_vl" in arch_blob:
            return Qwen25VLEmbeddingExtractor
        if "gemma4" in model_type or "gemma4" in arch_blob:
            return Gemma4EmbeddingExtractor
        if "phi4" in model_type or "phi4multimodal" in arch_blob.replace("_", ""):
            return Phi4EmbeddingExtractor
    except Exception:
        pass

    raise ValueError(
        f"Cannot resolve embedding extractor for model {model_name!r}. "
        "Supported families: Qwen2.5-VL, Gemma 4 VL, Phi-4 multimodal."
    )


class EmbeddingsExtractor:
    """
    Unified facade that delegates to the backend extractor chosen from ``model_name``.

    Same public API as ``Qwen25VLEmbeddingExtractor``: ``__init__``, ``extract``, ``close``.
    """

    def __init__(
        self,
        model_name: str,
        device=None,
        quantize_4_bit: bool = False,
        quantize_8_bit: bool = False,
        torch_dtype=None,
        system_prompt=None,
    ):
        self.model_name = model_name
        extractor_cls = resolve_extractor_class(model_name)
        self._extractor = extractor_cls(
            model_name=model_name,
            device=device,
            quantize_4_bit=quantize_4_bit,
            quantize_8_bit=quantize_8_bit,
            torch_dtype=torch_dtype,
            system_prompt=system_prompt,
        )

    @torch.no_grad()
    def extract(
        self,
        images: List[Image.Image],
        prompt: str = "Describe the images.",
    ) -> Dict[str, torch.Tensor]:
        return self._extractor.extract(images, prompt=prompt)

    def close(self) -> None:
        self._extractor.close()
