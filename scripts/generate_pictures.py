"Generates according solid rgb images for to all munsell colors from csv file"


import csv
import json
import os
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from utils.color.gen_color import munsell_csv_row_swatch
from utils.color.is_in_srgb import is_in_srgb_gamut


# Resolve project root and local_experiments path for imports and configs
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_FILE_DIR)
LOCAL_EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "local_experiments")

# Use config from the central configs folder
PICTURE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "picture_config.json")
SOURCE_CSV_PATH = os.path.join(LOCAL_EXPERIMENTS_DIR, "munsell_3-3.csv")

OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "munsell_colors")
OUTPUT_PICS_DIR = os.path.join(OUTPUT_BASE_DIR, "pics")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_BASE_DIR, "munsell_manifest.csv")


def load_picture_config(path: str) -> Tuple[int, int, str]:
    """
    Load picture size and color space from JSON config.
    Defaults: width=224, height=224, color_space='sRGB'.
    """
    width = 224
    height = 224
    color_space = "sRGB"
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            width = int(cfg.get("width", width))
            height = int(cfg.get("height", height))
            color_space = str(cfg.get("color_space", color_space))
    return width, height, color_space


def ensure_dirs():
    os.makedirs(OUTPUT_PICS_DIR, exist_ok=True)

def main():
    ensure_dirs()

    # Load picture configuration
    width, height, color_space = load_picture_config(PICTURE_CONFIG_PATH)

    # Read source CSV using pandas
    if not os.path.isfile(SOURCE_CSV_PATH):
        raise FileNotFoundError(f"Source CSV not found at {SOURCE_CSV_PATH}")

    df = pd.read_csv(SOURCE_CSV_PATH)

    # Ensure output columns exist and default to empty strings
    for col in ("picture", "R", "G", "B"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(object).where(pd.notna(df[col]), "")

    # Coerce necessary columns to numeric; non-numeric become NaN
    for col in ("x", "y", "Y"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select rows with valid numeric xyY
    valid_mask = df[["x", "y", "Y"]].notna().all(axis=1) if set(["x", "y", "Y"]).issubset(df.columns) else pd.Series(False, index=df.index)
    valid_indices = df.index[valid_mask]

    image_index = 1
    for idx in tqdm(valid_indices):
        x_val = float(df.at[idx, "x"])  # type: ignore[arg-type]
        y_val = float(df.at[idx, "y"])  # type: ignore[arg-type]
        Y_val = float(df.at[idx, "Y"])  # type: ignore[arg-type]

        # check if values xyY are in sRGB gamut
        if not is_in_srgb_gamut([x_val, y_val, Y_val]):
            continue

        filename = f"{image_index}.png"
        out_path = os.path.join(OUTPUT_PICS_DIR, filename)

        # CSV Y values are on ~0..100 scale, normalize by reference white Y=100.0
        img, srgb = munsell_csv_row_swatch(
            x_val,
            y_val,
            Y_val,
            size=(width, height),
            reference_white_Y=100.0,
            return_sRGB=True,
        )

        img.save(out_path)

        # Update DataFrame row
        df.at[idx, "picture"] = filename
        df.at[idx, "R"] = srgb[0]
        df.at[idx, "G"] = srgb[1]
        df.at[idx, "B"] = srgb[2]

        image_index += 1

    # Write augmented CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(
        f"Generated {image_index-1} images in '{OUTPUT_PICS_DIR}'. "
        f"Augmented CSV saved to '{OUTPUT_CSV_PATH}'. Color space: {color_space}."
    )


if __name__ == "__main__":
    main()
