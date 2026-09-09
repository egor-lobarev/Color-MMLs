"""Device selection helpers for embedding extractors."""

from __future__ import annotations

import warnings
from typing import Optional, Union

import torch


def cuda_is_usable() -> bool:
    """
    True only if CUDA can actually be initialized.

    ``torch.cuda.is_available()`` may be True while the installed PyTorch build
    still fails at runtime (e.g. NVIDIA driver older than the bundled CUDA).
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.mem_get_info(0)
        return True
    except RuntimeError:
        return False


def resolve_device(requested: Optional[str] = None) -> str:
    """
    Pick a runtime device string: ``"cuda"``, ``"cpu"``, or ``"mps"``.

    When ``requested`` is ``"cuda"`` but CUDA cannot be used, falls back to CPU
    with a warning instead of failing during ``from_pretrained``.
    """
    if requested is not None:
        key = requested.strip().lower()
        if key in ("cuda", "gpu"):
            if cuda_is_usable():
                return "cuda"
            warnings.warn(
                "CUDA was requested but is not usable (driver/PyTorch mismatch). "
                "Falling back to CPU.",
                stacklevel=2,
            )
            return "cpu"
        if key == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            warnings.warn("MPS was requested but is not available. Falling back to CPU.", stacklevel=2)
            return "cpu"
        return key

    return "cuda" if cuda_is_usable() else "cpu"


def default_dtype(device: str, torch_dtype=None) -> torch.dtype:
    """bfloat16 на CUDA, float32 иначе.

    Сравнение идёт по префиксу: конфиги задают устройство как ``"cuda:0"``, и
    строгое ``== "cuda"`` молча возвращало float32 — модель грузилась в двойном
    объёме памяти (InternVL3-14B ≈ 57 ГБ вместо 28, гарантированный OOM
    на A100 40 ГБ).
    """
    if torch_dtype is not None:
        return torch_dtype
    if str(device).split(":", 1)[0] in ("cuda", "gpu"):
        return torch.bfloat16
    return torch.float32


def hf_device_map(device: str) -> Union[str, dict]:
    """``device_map`` value for ``transformers.from_pretrained``."""
    if device == "cuda":
        return "auto"
    if device == "cpu":
        return "cpu"
    if device == "mps":
        return {"": "mps"}
    return device


def model_input_device(model: torch.nn.Module) -> torch.device:
    """Device where model weights live (works with accelerate ``device_map``)."""
    return next(model.parameters()).device
