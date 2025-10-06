from glob import glob
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple

import numpy as np
import torch
from PIL import Image

def _save_tensor_npy(path: Path, tensor: torch.Tensor):
    """
    Save a tensor as .npy (float32). Forces CPU + float to be portable and light for downstream use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.as_posix(), tensor.detach().cpu().float().numpy())


def gather_image_paths(paths: List[str], recursive: bool, valid_exts: List | Set | Tuple) -> List[Path]:
    cand: List[Path] = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in pth.glob(pattern):
                if f.is_file() and f.suffix.lower() in valid_exts:
                    cand.append(f)
        else:
            for m in glob(p, recursive=recursive):
                fp = Path(m)
                if fp.is_file() and fp.suffix.lower() in valid_exts:
                    cand.append(fp)
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for f in cand:
        r = f.resolve()
        if r not in seen:
            seen.add(r);
            uniq.append(Path(r))
    return uniq


def load_images(paths: List[Path]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs


def save_tensor(path: Path, tensor: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_tensor_npy(path, tensor)


def tensor_shape(t: Optional[torch.Tensor]):
    return None if t is None else tuple(t.shape)


def save_all(img_dir: Path, out: Dict[str, torch.Tensor], save_tokens: bool) -> Dict[str, str]:
    """Save pooled + (optionally) token-level + lens; return mapping of key->filepath."""
    pooled_keys = ("vision_pooled_mean", "projected_pooled_mean", "lm_pooled_mean")
    token_keys = ("vision_tokens", "projected_tokens", "lm_last_hidden")
    saved: Dict[str, str] = {}

    for k in pooled_keys:
        if k in out:
            save_tensor(img_dir / f"{k}.npy", out[k])
            saved[k] = (img_dir / f"{k}.npy").as_posix()

    if "visual_token_lens" in out:
        save_tensor(img_dir / "visual_token_lens.npy", out["visual_token_lens"])
        saved["visual_token_lens"] = (img_dir / "visual_token_lens.npy").as_posix()

    if save_tokens:
        for k in token_keys:
            if k in out:
                save_tensor(img_dir / f"{k}.npy", out[k])
                saved[k] = (img_dir / f"{k}.npy").as_posix()

    return saved
