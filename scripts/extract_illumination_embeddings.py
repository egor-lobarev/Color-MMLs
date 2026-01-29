"""
Extract embeddings for illumination estimation datasets.
Scans PNG/ subfolder for .JPG files and extracts VLM embeddings.
Saves outputs under embeddings/<experiment_name>/.

Usage:
    python scripts/extract_illumination_embeddings.py --config path/to/config.json

Config keys:
    - experiment_name: name for this experiment
    - data_path: path to dataset folder containing gt.csv and PNG/ subfolder
    - outdir_root: embeddings root (default: data/embeddings/qwen2.5_7B)
    - model, device, prompt, save_tokens, restart_model_per_image
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from utils.embeddings.embedding_extractor import Qwen25VLEmbeddingExtractor
from utils.embeddings.images_loader import load_images, save_all, tensor_shape


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings for illumination estimation datasets.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file with paths and model settings.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    print("Config:", cfg)

    experiment_name = cfg.get("experiment_name")
    if not experiment_name:
        raise SystemExit("Config must contain 'experiment_name' field.")
    
    data_path = Path(cfg.get("data_path")).expanduser()
    if not data_path.exists() or not data_path.is_dir():
        raise SystemExit(f"Data path not found or not a directory: {data_path.as_posix()}")

    # Check for PNG subfolder
    png_dir = data_path / "PNG"
    if not png_dir.exists() or not png_dir.is_dir():
        raise SystemExit(f"PNG subfolder not found in: {data_path.as_posix()}")

    # Find all .JPG files in PNG subfolder
    img_paths = sorted(png_dir.glob("*.JPG"))
    if not img_paths:
        raise SystemExit(f"No .JPG files found in: {png_dir.as_posix()}")

    print(f"Found {len(img_paths)} JPG files to process")

    # Setup output directory
    out_root_root = Path(cfg.get("outdir_root", "data/embeddings/qwen2.5_7B")).expanduser()
    out_root = out_root_root / experiment_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Load images
    images = load_images(img_paths)

    # Model settings
    model_name = cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
    device = cfg.get("device", None)
    prompt = cfg.get("prompt", "What is the illumination in this image?")
    save_tokens = bool(cfg.get("save_tokens", False))
    restart_model_per_image = bool(cfg.get("restart_model_per_image", False))
    init_prompt = cfg.get("init_prompt", None)

    extractor = None
    console = {"mode": "per-image", "items": []}

    for p, img in tqdm(zip(img_paths, images), total=len(img_paths), desc="Processing images"):
        stem = p.stem
        
        if restart_model_per_image:
            if extractor is not None:
                extractor.close()
                del extractor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            extractor = Qwen25VLEmbeddingExtractor(model_name=model_name, device=device)
        elif extractor is None:
            extractor = Qwen25VLEmbeddingExtractor(model_name=model_name, device=device, system_prompt=init_prompt)

        out = extractor.extract([img], prompt=prompt)
        img_dir = out_root / stem
        img_dir.mkdir(parents=True, exist_ok=True)
        saved = save_all(img_dir, out, save_tokens)

        # Save simple manifest without illumination metadata
        with open(img_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "image": p.as_posix(),
                "model": model_name,
                "init_prompt": init_prompt,
                "restart_model_per_image": restart_model_per_image,
                "prompt": prompt,
                "answer": out.get("model_answer", ""),
                "saved": saved,
                "shapes": {k: tensor_shape(out.get(k)) for k in
                           ("vision_pooled_mean","projected_pooled_mean","lm_pooled_mean",
                            "vision_tokens","projected_tokens","lm_last_hidden","visual_token_lens")},
            }, f, ensure_ascii=False, indent=2)

        console["items"].append({"image": p.as_posix(), "dir": img_dir.as_posix(), "saved": saved})

    print(json.dumps(console, indent=2))
    if extractor is not None:
        extractor.close()


if __name__ == "__main__":
    main()