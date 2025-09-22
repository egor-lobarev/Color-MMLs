"""
Extract embeddings for all images inside a color_dataset subfolder and save them
under embeddings/<subfolder_name>.

Run with config (no CLI args needed):
    python extract_color_dataset_embeddings.py

By default looks for local_experiments/embedding_config.json
You can override config path via: python extract_color_dataset_embeddings.py --config path/to/config.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from embedding_extractor import Qwen25VLEmbeddingExtractor
from utils import load_images, save_all, tensor_shape, save_tensor


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path.as_posix()}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings for a color_dataset subfolder (config-driven).")
    parser.add_argument("--config", type=str, default=str(Path("local_experiments/embedding_config.json")),
                        help="Path to JSON config file.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))

    dataset_dir = Path(cfg.get("dataset_dir", "")).expanduser()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found or not a directory: {dataset_dir.as_posix()}")

    subfolder_name = dataset_dir.name
    out_root_root = Path(cfg.get("outdir_root", "embeddings")).expanduser()
    out_root = out_root_root / subfolder_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Load dataset manifest to enrich embedding manifests with color metadata
    dataset_manifest_path = dataset_dir / "manifest.json"
    picture_size: Dict[str, int] = {}
    diagram_all_name: str = ""
    index_to_meta: Dict[str, Dict] = {}
    if dataset_manifest_path.exists():
        try:
            with open(dataset_manifest_path, "r", encoding="utf-8") as f:
                ds_manifest = json.load(f)
            # picture size
            if isinstance(ds_manifest.get("picture_size"), dict):
                picture_size = {
                    "width": ds_manifest["picture_size"].get("width"),
                    "height": ds_manifest["picture_size"].get("height"),
                }
            # diagram image file to skip
            if isinstance(ds_manifest.get("diagram_all"), str):
                diagram_all_name = ds_manifest["diagram_all"]
            # Build index -> {notation, xyY}
            for chain in ds_manifest.get("chains", []):
                for item in chain.get("items", []):
                    idx = str(item.get("index"))
                    if not idx or idx == "None":
                        continue
                    index_to_meta[idx] = {
                        "notation": item.get("notation"),
                        "xyY": item.get("xyY"),
                    }
        except Exception:
            # If dataset manifest is malformed, proceed without enrichment
            picture_size = {}
            diagram_all_name = ""
            index_to_meta = {}

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    # Collect images from the dataset subfolder (non-recursive)
    img_paths: List[Path] = [
        p for p in sorted(dataset_dir.iterdir())
        if p.suffix.lower() in valid_exts
        and p.name != "chains_xy.png"
        and (not diagram_all_name or p.name != diagram_all_name)
    ]
    if not img_paths:
        raise SystemExit("No images found in dataset folder. Supported: " + ", ".join(sorted(valid_exts)))

    print("Found images to process:", [p.as_posix() for p in img_paths])
    images = load_images(img_paths)
    stems = [p.stem for p in img_paths]

    model_name = cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
    device = cfg.get("device", None)
    prompt = cfg.get("prompt", "Describe the image(s).")
    save_tokens = bool(cfg.get("save_tokens", False))
    batch = bool(cfg.get("batch", False))
    lm_per_image = bool(cfg.get("lm_per_image", False))
    restart_model_per_image = bool(cfg.get("restart_model_per_image", False))

    extractor = None

    console = {"mode": "batch" if batch else "per-image", "items": []}

    # Per-image mode
    if not batch:
        for p, stem, img in tqdm(zip(img_paths, stems, images), total=len(img_paths), desc="Processing images"):
            # Create fresh extractor for each image if requested
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

            with open(img_dir / "manifest.json", "w", encoding="utf-8") as f:
                meta = index_to_meta.get(stem, {})
                json.dump({
                    "image": p.as_posix(),
                    "model": model_name,
                    "prompt": prompt,
                    "answer": out.get("model_answer", ""),
                    "saved": saved,
                    "shapes": {k: tensor_shape(out.get(k)) for k in
                               ("vision_pooled_mean","projected_pooled_mean","lm_pooled_mean",
                                "vision_tokens","projected_tokens","lm_last_hidden","visual_token_lens")},
                    "picture_size": picture_size or None,
                    "munsell_spec": meta.get("notation"),
                    "xyY": meta.get("xyY"),
                }, f, ensure_ascii=False, indent=2)

            console["items"].append({"image": p.as_posix(), "dir": (out_root / stem).as_posix(), "saved": saved})

        print(json.dumps(console, indent=2))
        if extractor is not None:
            extractor.close()
        return

    # Batch mode
    if extractor is None:
        extractor = Qwen25VLEmbeddingExtractor(model_name=model_name, device=device)
    out_batch = extractor.extract(images, prompt=prompt)
    batch_dir = out_root / f"batch_{len(images)}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save global sequence-level artifacts
    global_saved: Dict[str, str] = {}
    if "lm_pooled_mean" in out_batch:
        save_tensor(batch_dir / "lm_pooled_mean.npy", out_batch["lm_pooled_mean"])
        global_saved["lm_pooled_mean"] = (batch_dir / "lm_pooled_mean.npy").as_posix()
    if save_tokens and "lm_last_hidden" in out_batch:
        save_tensor(batch_dir / "lm_last_hidden.npy", out_batch["lm_last_hidden"])
        global_saved["lm_last_hidden"] = (batch_dir / "lm_last_hidden.npy").as_posix()
    if "visual_token_lens" in out_batch:
        save_tensor(batch_dir / "visual_token_lens.npy", out_batch["visual_token_lens"])
        global_saved["visual_token_lens"] = (batch_dir / "visual_token_lens.npy").as_posix()

    # Prepare per-image splits for vision/projected tokens
    lens = out_batch.get("visual_token_lens", None)
    vtoks = out_batch.get("vision_tokens", None)
    ptoks = out_batch.get("projected_tokens", None)

    need_fallback = lens is None
    idxs = None
    if lens is not None:
        lengths = lens.detach().cpu().tolist()
        csum = [0]
        for L in lengths:
            csum.append(csum[-1] + int(L))
        idxs = list(zip(csum[:-1], csum[1:]))

    for i, (p, stem) in enumerate(tqdm(zip(img_paths, stems), total=len(img_paths), desc="Processing batch images")):
        img_dir = batch_dir / stem
        img_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        answer_i = None

        if not need_fallback and vtoks is not None:
            s, e = idxs[i]
            if e > s:
                v_slice = vtoks[s:e]
                v_pool = v_slice.mean(dim=0, keepdim=True)
                save_tensor(img_dir / "vision_pooled_mean.npy", v_pool)
                saved["vision_pooled_mean"] = (img_dir / "vision_pooled_mean.npy").as_posix()
                if save_tokens:
                    save_tensor(img_dir / "vision_tokens.npy", v_slice)
                    saved["vision_tokens"] = (img_dir / "vision_tokens.npy").as_posix()

        if not need_fallback and ptoks is not None:
            s, e = idxs[i]
            if e > s:
                p_slice = ptoks[s:e]
                p_pool = p_slice.mean(dim=0, keepdim=True)
                save_tensor(img_dir / "projected_pooled_mean.npy", p_pool)
                saved["projected_pooled_mean"] = (img_dir / "projected_pooled_mean.npy").as_posix()
                if save_tokens:
                    save_tensor(img_dir / "projected_tokens.npy", p_slice)
                    saved["projected_tokens"] = (img_dir / "projected_tokens.npy").as_posix()

        # If lens is missing OR user wants per-image LLM pools, do a quick single-image pass
        if need_fallback or lm_per_image:
            # Create fresh extractor for each image if requested
            if restart_model_per_image:
                if extractor is not None:
                    extractor.close()
                    del extractor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                extractor = Qwen25VLEmbeddingExtractor(model_name=model_name, device=device)
            elif extractor is None:
                extractor = Qwen25VLEmbeddingExtractor(model_name=model_name, device=device)
            
            out_i = extractor.extract([images[i]], prompt=prompt)
            if need_fallback:
                if "vision_pooled_mean" in out_i and "vision_pooled_mean" not in saved:
                    save_tensor(img_dir / "vision_pooled_mean.npy", out_i["vision_pooled_mean"])
                    saved["vision_pooled_mean"] = (img_dir / "vision_pooled_mean.npy").as_posix()
                if "projected_pooled_mean" in out_i and "projected_pooled_mean" not in saved:
                    save_tensor(img_dir / "projected_pooled_mean.npy", out_i["projected_pooled_mean"])
                    saved["projected_pooled_mean"] = (img_dir / "projected_pooled_mean.npy").as_posix()
                if save_tokens:
                    for k in ("vision_tokens", "projected_tokens"):
                        if k in out_i and k not in saved:
                            save_tensor(img_dir / f"{k}.npy", out_i[k])
                            saved[k] = (img_dir / f"{k}.npy").as_posix()
            if "lm_pooled_mean" in out_i:
                save_tensor(img_dir / "lm_pooled_mean.npy", out_i["lm_pooled_mean"])
                saved["lm_pooled_mean"] = (img_dir / "lm_pooled_mean.npy").as_posix()
            if save_tokens and "lm_last_hidden" in out_i:
                save_tensor(img_dir / "lm_last_hidden.npy", out_i["lm_last_hidden"])
                saved["lm_last_hidden"] = (img_dir / "lm_last_hidden.npy").as_posix()
            answer_i = out_i.get("model_answer", "")

        with open(img_dir / "manifest.json", "w", encoding="utf-8") as f:
            meta = index_to_meta.get(stem, {})
            json.dump({
                "image": p.as_posix(),
                "model": model_name,
                "prompt": prompt,
                "answer": answer_i,
                "saved": saved,
                "picture_size": picture_size or None,
                "munsell_spec": meta.get("notation"),
                "xyY": meta.get("xyY"),
            }, f, ensure_ascii=False, indent=2)

        console["items"].append({"image": p.as_posix(), "dir": img_dir.as_posix(), "saved": saved})

    with open(batch_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "batch_size": len(images),
            "model": model_name,
            "prompt": prompt,
            "batch_answer": out_batch.get("model_answer", ""),
            "global_saved": global_saved,
            "images": [p.as_posix() for p in img_paths],
        }, f, ensure_ascii=False, indent=2)

    console["batch_dir"] = batch_dir.as_posix()
    console["global_saved"] = global_saved
    print(json.dumps(console, indent=2))
    if extractor is not None:
        extractor.close()


if __name__ == "__main__":
    main()


