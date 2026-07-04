"""
compress_embeddings.py

For each leaf folder under <root> that contains projected_pooled_mean.npy
and vision_pooled_mean.npy:

  1. Move the originals into a subfolder called "tokens/"
  2. Average projected_pooled_mean over the token dimension → shape [N, 1, C]
     and save as projected_pooled_mean.npy  (next to manifest.json)
  3. Average vision_pooled_mean  over the token dimension → shape [N, 1, C]
     and save as vision_pooled_mean.npy

Usage:
    python compress_embeddings.py /path/to/embeddings/InternVL3-2B
    python compress_embeddings.py /path/to/embeddings/InternVL3-2B --dry-run
    python compress_embeddings.py /path/to/embeddings/InternVL3-2B --skip-existing
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


KEYS_TO_COMPRESS = ["projected_pooled_mean", "vision_pooled_mean"]


def find_leaf_dirs(root: Path) -> list[Path]:
    """Return all directories that contain at least one of the target .npy files."""
    return sorted(
        {p.parent for key in KEYS_TO_COMPRESS for p in root.rglob(f"{key}.npy")}
    )


def compress_dir(folder: Path, dry_run: bool, skip_existing: bool) -> dict:
    tokens_dir = folder / "tokens"
    result = {"folder": str(folder), "status": "ok", "skipped": [], "errors": []}

    for key in KEYS_TO_COMPRESS:
        src = folder / f"{key}.npy"
        if not src.exists():
            result["skipped"].append(key)
            continue

        dst_tokens = tokens_dir / f"{key}.npy"
        dst_compressed = folder / f"{key}.npy"   # same name, replaced in-place

        # --- check if already done ---
        if skip_existing and dst_tokens.exists():
            result["skipped"].append(f"{key} (tokens already exist)")
            continue

        try:
            arr = np.load(src)                   # e.g. [N_tokens, C] or [1, N_tokens, C]
        except Exception as e:
            result["errors"].append(f"{key}: load failed — {e}")
            continue

        original_shape = arr.shape

        # Normalize to [B, N_tokens, C]
        if arr.ndim == 2:
            arr = arr[np.newaxis]                # [N, C] → [1, N, C]

        if arr.ndim != 3:
            result["errors"].append(
                f"{key}: unexpected shape {original_shape}, skipping"
            )
            continue

        # Mean over token dimension → [B, 1, C]
        compressed = arr.mean(axis=1, keepdims=True)

        print(f"  {key}: {original_shape} → compressed {compressed.shape}")

        if dry_run:
            print(f"    [dry-run] would move   {src} → {dst_tokens}")
            print(f"    [dry-run] would write  {dst_compressed}  ({compressed.nbytes/1024:.1f} KiB)")
            continue

        # Move original to tokens/
        tokens_dir.mkdir(exist_ok=True)
        shutil.move(str(src), str(dst_tokens))

        # Save compressed in-place
        np.save(dst_compressed, compressed)

    return result


def main():
    parser = argparse.ArgumentParser(description="Compress pooled embedding files.")
    parser.add_argument("root", type=Path, help="Root embeddings directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without changing anything"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip folders where tokens/ subdirectory already exists"
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"ERROR: {args.root} is not a directory", file=sys.stderr)
        sys.exit(1)

    folders = find_leaf_dirs(args.root)
    if not folders:
        print(f"No {KEYS_TO_COMPRESS} files found under {args.root}")
        sys.exit(0)

    print(f"Found {len(folders)} folder(s) to process under {args.root}")
    if args.dry_run:
        print("--- DRY RUN — nothing will be written ---\n")

    n_ok = n_skip = n_err = 0
    for folder in folders:
        print(f"\n[{folder.relative_to(args.root)}]")
        r = compress_dir(folder, dry_run=args.dry_run, skip_existing=args.skip_existing)

        if r["skipped"]:
            print(f"  skipped: {', '.join(r['skipped'])}")
            n_skip += 1
        if r["errors"]:
            for e in r["errors"]:
                print(f"  ERROR: {e}", file=sys.stderr)
            n_err += 1
        if not r["errors"] and not (r["skipped"] and not args.dry_run):
            n_ok += 1

    print(f"\nDone — {n_ok} compressed, {n_skip} skipped, {n_err} errors")


if __name__ == "__main__":
    main()