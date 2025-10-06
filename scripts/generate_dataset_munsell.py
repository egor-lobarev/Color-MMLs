import json
import os
from typing import Dict, List

import numpy as np

import matplotlib.pyplot as plt
import colour
import colour.plotting as cp
from matplotlib import cm, colors as mcolors

from utils.color.gen_color import MunsellColor, munsell_swatch

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "color_dataset", "black")
PICTURE_CFG_PATH = os.path.join(HERE, "picture_config.json")
CHAINS_CFG_PATH = os.path.join(HERE, "black.json")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")


def load_picture_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chains_config(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["chains"]


def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)


def _munsell_spec_from_color(col: MunsellColor) -> str:
    if isinstance(col.h, str) and col.h.upper() == "N":
        return f"N{col.v}"
    return f"{col.h} {col.v}/{col.c}"


def is_in_srgb_gamut(munsell_spec: str) -> bool:
    """
    Check if given Munsell color is within the sRGB gamut by converting to linear sRGB
    (no encoding) and verifying all channels are in [0, 1] without clipping.
    """
    try:
        xyY = colour.munsell_colour_to_xyY(munsell_spec)
        xyY = np.array(xyY, dtype=float)
        XYZ = colour.xyY_to_XYZ(xyY)
        # Linear sRGB
        rgb_linear = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)
        return np.all(rgb_linear >= 0.0) and np.all(rgb_linear <= 1.0)
    except Exception:
        return False


def build_chain(desc: Dict) -> List[MunsellColor]:
    """
    Build a chain of MunsellColor from a chain descriptor:
    {
      "variable": "c" | "v" | "h",
      "values": [..],
      "fixed": {"h": "2.5R", "c": 6, "v": 5}
    }
    """
    variable = desc["variable"].lower()
    values = desc["values"]
    fixed = desc["fixed"]
    fixed_h = fixed.get("h")
    fixed_c = fixed.get("c")
    fixed_v = fixed.get("v")

    return MunsellColor.chain(
        variable, values, fixed_h=fixed_h, fixed_c=fixed_c, fixed_v=fixed_v
    )


def main():
    ensure_dirs()

    pic_cfg = load_picture_config(PICTURE_CFG_PATH)
    width = int(pic_cfg["width"])  # required
    height = int(pic_cfg["height"])  # required
    color_space = str(pic_cfg.get("color_space", "sRGB"))

    chains = load_chains_config(CHAINS_CFG_PATH)

    manifest = {
        "picture_size": {"width": width, "height": height},
        "color_space": color_space,
        "chains": [],
    }

    img_index = 1
    all_xy_groups = []  # list of (xs, ys, label)
    for chain_idx, ch in enumerate(chains, start=1):
        # Create colors in this chain
        colors = build_chain(ch)

        fixed = ch["fixed"]
        fixed_desc = (
            f"chain fixed H={fixed.get('h')} C={fixed.get('c')} V={fixed.get('v')}"
        )

        items_meta = []
        mapping_num_to_varvalue = {}

        # Collect xyY for plotting (per-chain)
        xy_points = []  # list of (x, y, Yref)

        for col in colors:
            # Build Munsell notation and gamut check
            # Handle neutral: chroma is not applicable and some V might be unsupported
            if isinstance(col.h, str) and col.h.upper() == 'N':
                # Skip chains varying chroma when hue is neutral
                if ch["variable"].lower() == 'c':
                    continue

                # Force chroma to 0 in representation/manifest, and build neutral notation
                col = MunsellColor(h='N', c=0, v=int(col.v))
            munsell_str = _munsell_spec_from_color(col)
            if not is_in_srgb_gamut(munsell_str):
                continue
            try:
                img = munsell_swatch(munsell_str, (width, height))
            except (ValueError, AssertionError) as exc:
                print(munsell_str, "is not valid")
                print(exc)
                continue
            out_path = os.path.join(DATASET_DIR, f"{img_index}.png")
            img.save(out_path)

            # xyY for diagram and manifest (Y is reference scale ~0..100)
            xyY = None
            try:
                xyY = colour.munsell_colour_to_xyY(munsell_str)
                x_val, y_val, Y_val = float(xyY[0]), float(xyY[1]), float(xyY[2])
                xy_points.append((x_val, y_val, Y_val))
            except Exception:
                x_val = y_val = Y_val = None

            # Determine variable value
            if ch["variable"].lower() == "h":
                var_val = col.h
            elif ch["variable"].lower() == "c":
                var_val = col.c
            else:
                var_val = col.v

            items_meta.append(
                {
                    "index": img_index,
                    "H": col.h,
                    "C": col.c,
                    "V": col.v,
                    "notation": munsell_str,
                    "xyY": {
                        "x": x_val,
                        "y": y_val,
                        "Y": Y_val,
                    },
                }
            )
            mapping_num_to_varvalue[str(img_index)] = var_val

            img_index += 1

        manifest["chains"].append(
            {
                "description": fixed_desc,
                "variable": ch["variable"],
                "fixed": ch["fixed"],
                "items": items_meta,
                "map_picture_to_variable_value": mapping_num_to_varvalue,
            }
        )

        # Keep data for combined diagram
        if xy_points:
            xs, ys, Ys = zip(*xy_points)
            label = f"Chain {chain_idx} ({ch['variable']})"
            all_xy_groups.append((list(xs), list(ys), list(Ys), label))

    # Plot and save a single combined xy chromaticity diagram with legend
    if all_xy_groups:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        cp.plot_chromaticity_diagram_CIE1931(
            cmfs='CIE 1931 2 Degree Standard Observer',
            show=False,
            axes=ax,
            bounding_box=[0, 0.8, 0, 0.9],
            transparent_background=True,
            diagram_opacity=0.5,
        )
        # Cycle through markers; use colormap for Y reference luminance
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        for i, (xs, ys, Ys, label) in enumerate(all_xy_groups):
            ax.scatter(
                xs,
                ys,
                s=25,
                marker=markers[i % len(markers)],
                c=Ys,
                cmap=cmap,
                norm=norm,
                label=label,
                edgecolors='none',
            )
        # Overlay sRGB triangle (primaries in xy, D65 white)
        try:
            cs = colour.models.RGB_COLOURSPACES['sRGB']
            prim = cs.primaries  # [[xR,yR],[xG,yG],[xB,yB]]
            tri_x = [prim[0,0], prim[1,0], prim[2,0], prim[0,0]]
            tri_y = [prim[0,1], prim[1,1], prim[2,1], prim[0,1]]
            ax.plot(tri_x, tri_y, color='tab:red', linewidth=1.5, label='sRGB gamut')
        except Exception:
            pass
        # Add colorbar for Y (reference, 0..1)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Y (reference)')
        ax.legend(loc='upper right', fontsize=8, frameon=True)
        fig.dpi=300
        ax.set_title("All Chains: CIE 1931 xy Chromaticity Diagram")
        diagram_all_path = os.path.join(DATASET_DIR, "chains_xy.png")
        plt.tight_layout()
        fig.savefig(diagram_all_path, dpi=300)
        plt.close(fig)
        manifest["diagram_all"] = os.path.basename(diagram_all_path)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
