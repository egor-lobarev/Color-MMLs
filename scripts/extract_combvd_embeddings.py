"""
Extract Qwen2.5-VL embeddings for COMBVD image folders.

Expected input layout (from scripts/generate_combvd_pictures.py):
    data/colors/combvd/<dataset_name>/*.png
    data/colors/combvd/<dataset_name>/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from utils.embeddings.embedding_extractor import Qwen25VLEmbeddingExtractor
from utils.embeddings.images_loader import load_images, save_all, tensor_shape


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path.as_posix()}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sorted_images(dataset_dir: Path) -> List[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    candidates = [p for p in dataset_dir.iterdir() if p.suffix.lower() in valid_exts]

    def key_fn(p: Path):
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(candidates, key=key_fn)


def _load_index_meta(dataset_dir: Path) -> Dict[str, Dict]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    index_meta: Dict[str, Dict] = {}
    for item in items:
        filename = str(item.get("file", ""))
        stem = Path(filename).stem
        index_meta[stem] = item
    return index_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings for all COMBVD dataset folders."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("configs/combvd_embeddings.json")),
        help="Path to JSON config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(Path(args.config))

    dataset_root = Path(cfg.get("dataset_root", "data/colors/combvd")).expanduser()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise SystemExit(
            f"Dataset root not found or not a directory: {dataset_root.as_posix()}"
        )

    requested_datasets: Optional[List[str]] = cfg.get("datasets")
    if requested_datasets:
        dataset_dirs = [
            dataset_root / name for name in requested_datasets if (dataset_root / name).is_dir()
        ]
    else:
        dataset_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    if not dataset_dirs:
        raise SystemExit(f"No dataset subfolders found in: {dataset_root.as_posix()}")

    out_root_root = Path(cfg.get("outdir_root", "data/embeddings/qwen2.5_7B/combvd")).expanduser()
    out_root_root.mkdir(parents=True, exist_ok=True)

    model_name = cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
    device = cfg.get("device", None)
    prompt = cfg.get("prompt", "Describe the image(s).")
    save_tokens = bool(cfg.get("save_tokens", False))
    restart_model_per_image = bool(cfg.get("restart_model_per_image", False))
    init_prompt = cfg.get("init_prompt", None)

    extractor: Optional[Qwen25VLEmbeddingExtractor] = None

    console: Dict[str, object] = {
        "dataset_root": dataset_root.as_posix(),
        "out_root_root": out_root_root.as_posix(),
        "datasets": [],
    }

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        img_paths = _sorted_images(dataset_dir)
        if not img_paths:
            print(f"Skipping {dataset_name}: no images found.")
            continue

        dataset_out_root = out_root_root / dataset_name
        dataset_out_root.mkdir(parents=True, exist_ok=True)
        index_to_meta = _load_index_meta(dataset_dir)
        images = load_images(img_paths)

        ds_console = {"dataset": dataset_name, "items": []}

        for p, img in tqdm(
            zip(img_paths, images),
            total=len(img_paths),
            desc=f"Processing {dataset_name}",
        ):
            if restart_model_per_image:
                if extractor is not None:
                    extractor.close()
                    del extractor
                    extractor = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if extractor is None:
                extractor = Qwen25VLEmbeddingExtractor(
                    model_name=model_name,
                    device=device,
                    system_prompt=init_prompt,
                )

            out = extractor.extract([img], prompt=prompt)
            stem = p.stem
            img_dir = dataset_out_root / stem
            img_dir.mkdir(parents=True, exist_ok=True)
            saved = save_all(img_dir, out, save_tokens)

            meta = index_to_meta.get(stem, {})
            with open(img_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": dataset_name,
                        "image": p.as_posix(),
                        "model": model_name,
                        "init_prompt": init_prompt,
                        "prompt": prompt,
                        "answer": out.get("model_answer", ""),
                        "saved": saved,
                        "shapes": {
                            k: tensor_shape(out.get(k))
                            for k in (
                                "vision_pooled_mean",
                                "projected_pooled_mean",
                                "lm_pooled_mean",
                                "vision_tokens",
                                "projected_tokens",
                                "lm_last_hidden",
                                "visual_token_lens",
                            )
                        },
                        "index": meta.get("index"),
                        "xyz": meta.get("xyz"),
                        "xyY": meta.get("xyY"),
                        "srgb": meta.get("srgb"),
                        "rgb_255": meta.get("rgb_255"),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            ds_console["items"].append(
                {"image": p.as_posix(), "dir": img_dir.as_posix(), "saved": saved}
            )

        console["datasets"].append(ds_console)

    if extractor is not None:
        extractor.close()

    print(json.dumps(console, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
