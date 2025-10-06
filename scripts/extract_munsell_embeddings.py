"""
Extract embeddings for all colors listed in a CSV manifest (pictures directory + paths
provided via a JSON config). Saves outputs under embeddings/<csv_parent_dir_name>.

Expected CSV columns (from scripts/generate_pictures.py):
    H,V,C,x,y,Y,picture[,R,G,B]

Usage (single parameter only):
    python scripts/extract_munsell_embeddings.py --config path/to/config.json

Config keys:
    - csv: path to CSV manifest (default: data/munsell_colors/munsell_manifest.csv)
    - pics_dir: directory with color pictures (default: data/munsell_colors/pics)
    - outdir_root: embeddings root (default: embeddings)
    - model, device, prompt, save_tokens, restart_model_per_image
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
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
    parser = argparse.ArgumentParser(description="Extract embeddings for colors listed in a CSV manifest (config-driven).")
    parser.add_argument("--config", type=str, default=str(Path("local_experiments/embedding_config.json")),
                        help="Path to JSON config file with paths and model settings.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))

    csv_path = Path(cfg.get("csv", "data/munsell_colors/munsell_manifest.csv")).expanduser()
    pics_dir = Path(cfg.get("pics_dir", "data/munsell_colors/pics")).expanduser()
    if not csv_path.exists():
        raise SystemExit(f"CSV manifest not found: {csv_path.as_posix()}")
    if not pics_dir.exists() or not pics_dir.is_dir():
        raise SystemExit(f"Pics directory not found or not a directory: {pics_dir.as_posix()}")

    subfolder_name = csv_path.parent.name
    out_root_root = Path(cfg.get("outdir_root", "embeddings")).expanduser()
    out_root = out_root_root / subfolder_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Read CSV with pandas and build image list
    df = pd.read_csv(csv_path)
    if "picture" not in df.columns:
        raise SystemExit("CSV manifest must contain 'picture' column.")
    df["picture"] = df["picture"].astype(str).str.strip()
    df = df[df["picture"] != ""]
    if df.empty:
        raise SystemExit("No valid rows with 'picture' found in CSV manifest.")

    img_paths: List[Path] = []
    stems: List[str] = []
    index_to_meta: Dict[str, Dict] = {}
    for row in df.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row._asdict())  # type: ignore[attr-defined]
        pic_name = str(row_dict.get("picture", "")).strip()
        p = pics_dir / pic_name
        if not p.exists():
            continue
        img_paths.append(p)
        stem = p.stem
        stems.append(stem)
        index_to_meta[stem] = {
            "csv_row": row_dict,
            "munsell_spec": row_dict.get("H"),
            "xyY": {"x": row_dict.get("x"), "y": row_dict.get("y"), "Y": row_dict.get("Y")},
            "RGB": {"R": row_dict.get("R"), "G": row_dict.get("G"), "B": row_dict.get("B")},
        }

    if not img_paths:
        raise SystemExit("No images found corresponding to CSV 'picture' column.")

    print("Found images to process:", [p.as_posix() for p in img_paths])
    images = load_images(img_paths)

    model_name = cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
    device = cfg.get("device", None)
    prompt = cfg.get("prompt", "Describe the image(s).")
    save_tokens = bool(cfg.get("save_tokens", False))
    restart_model_per_image = bool(cfg.get("restart_model_per_image", False))

    extractor = None
    console = {"mode": "per-image", "items": []}

    for p, stem, img in tqdm(zip(img_paths, stems, images), total=len(img_paths), desc="Processing images"):
        if restart_model_per_image:
            if extractor is not None:
                extractor.close()
                del extractor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            extractor = Qwen25VLEmbeddingExtractor(model_name=model_name, device=device)
        elif extractor is None:
            extractor = Qwen25VLEmbeddingExtractor(model_name=model_name, device=device)

        out = extractor.extract([img], prompt=prompt)
        img_dir = out_root / stem
        img_dir.mkdir(parents=True, exist_ok=True)
        saved = save_all(img_dir, out, save_tokens)

        meta = index_to_meta.get(stem, {})
        with open(img_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "image": p.as_posix(),
                "model": model_name,
                "prompt": prompt,
                "answer": out.get("model_answer", ""),
                "saved": saved,
                "shapes": {k: tensor_shape(out.get(k)) for k in
                           ("vision_pooled_mean","projected_pooled_mean","lm_pooled_mean",
                            "vision_tokens","projected_tokens","lm_last_hidden","visual_token_lens")},
                "csv_row": meta.get("csv_row"),
                "munsell_spec": meta.get("munsell_spec"),
                "xyY": meta.get("xyY"),
                "RGB": meta.get("RGB"),
            }, f, ensure_ascii=False, indent=2)

        console["items"].append({"image": p.as_posix(), "dir": img_dir.as_posix(), "saved": saved})

    print(json.dumps(console, indent=2))
    if extractor is not None:
        extractor.close()


if __name__ == "__main__":
    main()


