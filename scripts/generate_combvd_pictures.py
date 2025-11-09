"""
Generate solid-color PNG images for COMBVD datasets from XYZ values.

Usage:
    python scripts/generate_combvd_pictures.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm


def _import_load_combvd():
    """
    Import COMBVD loader from installed vsl_ial; fallback to local vsl_ial_debug.
    """
    try:
        from vsl_ial.datasets.distance import load_combvd  # type: ignore

        return load_combvd
    except ModuleNotFoundError:
        import sys

        project_root = Path(__file__).resolve().parent.parent
        local_vsl = project_root / "vsl_ial_debug"
        sys.path.insert(0, local_vsl.as_posix())
        from vsl_ial.datasets.distance import load_combvd  # type: ignore

        return load_combvd


_XYZ_TO_LINRGB = np.array(
    (
        (3.2404542, -0.9692660, 0.0556434),
        (-1.5371385, 1.8760108, -0.2040259),
        (-0.4985314, 0.0415560, 1.0572252),
    ),
    dtype=np.float64,
)


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ (0..1 scale) to display-ready sRGB (0..1), with clipping.
    """
    lin_rgb = xyz @ _XYZ_TO_LINRGB
    lin_rgb = np.clip(lin_rgb, 0.0, 1.0)

    threshold = 0.0031308
    a = 0.055
    srgb = np.where(
        lin_rgb <= threshold,
        lin_rgb * 12.92,
        (1.0 + a) * np.power(lin_rgb, 1.0 / 2.4) - a,
    )
    return np.clip(srgb, 0.0, 1.0)


def xyz_to_xyy(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to xyY. Input shape: [..., 3], output shape: [..., 3].
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    X = xyz[..., 0]
    Y = xyz[..., 1]
    Z = xyz[..., 2]
    denom = X + Y + Z
    x = np.divide(X, denom, out=np.zeros_like(X), where=denom > 0)
    y = np.divide(Y, denom, out=np.zeros_like(Y), where=denom > 0)
    return np.stack([x, y, Y], axis=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate COMBVD color images from XYZ values."
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/colors/combvd",
        help="Output directory for generated datasets.",
    )
    parser.add_argument("--width", type=int, default=224, help="Image width.")
    parser.add_argument("--height", type=int, default=224, help="Image height.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset names to generate (e.g. witt leeds).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    load_combvd = _import_load_combvd()
    datasets = load_combvd()
    if args.datasets:
        allowed = set(args.datasets)
        datasets = [ds for ds in datasets if ds.name in allowed]
        if not datasets:
            raise SystemExit(
                "No COMBVD datasets matched --datasets: " + ", ".join(args.datasets)
            )

    global_manifest: Dict[str, object] = {
        "source": "vsl_ial.datasets.distance.load_combvd",
        "picture_size": {"width": args.width, "height": args.height},
        "datasets": [],
    }

    for ds in datasets:
        ds_dir = out_root / ds.name
        ds_dir.mkdir(parents=True, exist_ok=True)

        items: List[Dict[str, object]] = []
        csv_rows: List[List[object]] = []
        for idx, xyz in tqdm(
            enumerate(ds.xyz),
            total=len(ds.xyz),
            desc=f"Generating {ds.name}",
        ):
            srgb = xyz_to_srgb(np.asarray(xyz, dtype=np.float64).reshape(1, 3))[0]
            xyy = xyz_to_xyy(np.asarray(xyz, dtype=np.float64).reshape(1, 3))[0]
            rgb8 = np.round(srgb * 255.0).astype(np.uint8)
            image = Image.new("RGB", (args.width, args.height), tuple(rgb8.tolist()))
            filename = f"{idx}.png"
            image.save(ds_dir / filename)

            items.append(
                {
                    "index": idx,
                    "file": filename,
                    "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "xyY": [float(xyy[0]), float(xyy[1]), float(xyy[2])],
                    "srgb": [float(srgb[0]), float(srgb[1]), float(srgb[2])],
                    "rgb_255": [int(rgb8[0]), int(rgb8[1]), int(rgb8[2])],
                }
            )
            csv_rows.append(
                [
                    "",  # H (not applicable for COMBVD)
                    "",  # V
                    "",  # C
                    float(xyy[0]),
                    float(xyy[1]),
                    float(xyy[2]),
                    filename,
                    int(rgb8[0]),
                    int(rgb8[1]),
                    int(rgb8[2]),
                ]
            )

        ds_manifest = {
            "dataset": ds.name,
            "num_colors": len(ds.xyz),
            "num_pairs": len(ds.pairs),
            "pairs": [list(p) for p in ds.pairs],
            "L_A": float(ds.L_A),
            "Y_b": float(ds.Y_b),
            "c": float(ds.c),
            "Nc": float(ds.Nc),
            "F": float(ds.F),
            "illuminant": [float(v) for v in ds.illuminant.tolist()],
            "picture_size": {"width": args.width, "height": args.height},
            "items": items,
        }
        with open(ds_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(ds_manifest, f, ensure_ascii=False, indent=2)

        csv_path = ds_dir / "manifest.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["H", "V", "C", "x", "y", "Y", "picture", "R", "G", "B"])
            writer.writerows(csv_rows)

        global_manifest["datasets"].append(
            {
                "dataset": ds.name,
                "dir": ds_dir.as_posix(),
                "csv": csv_path.as_posix(),
                "num_colors": len(ds.xyz),
                "num_pairs": len(ds.pairs),
            }
        )

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(global_manifest, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "out_root": out_root.as_posix(),
                "datasets": len(datasets),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
