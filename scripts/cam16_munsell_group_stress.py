"""
CAM16 STRESS on Munsell chains with GLOBAL-k vs GROUP-k (per H/C/V).

Motivation: the learned linear map is evaluated with per-group scaling (Group
STRESS 0.15-0.17), but the CAM16 baseline was reported only with a single global
k (STRESS 0.548). That is not a like-for-like comparison. Here CAM16 gets the same
treatment: separate optimal k and separate STRESS for the varying-H, varying-C and
varying-V chains, plus their mean (Group STRESS).

Target: neighboring colors in a uniform Munsell chain are 1 perceptual step apart.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from colour import xyY_to_XYZ
from colour.models import XYZ_to_CAM16UCS, XYZ_to_CAM16LCD, XYZ_to_CAM16SCD
from colour.difference import (delta_E_CAM16UCS, delta_E_CAM16LCD,
                               delta_E_CAM16SCD)
from colour import XYZ_to_Lab
from vsl_ial.stress import stress as stress_fn

# NB: colour-science does NOT bake K_L into the UCS/LCD/SCD coordinates, so a
# plain Euclidean norm on the coordinates understates/overstates the lightness
# weighting for K_L != 1 (LCD: 0.77, SCD: 1.24). Formula-proper distances must
# go through the delta_E_* functions. UCS has K_L = 1, so both agree there.

CSV = "data/munsell_3-3.csv"

# standard Munsell hue order: 10 families x {2.5,5,7.5,10}
FAMILIES = ["R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"]
STEP = {"2.5": 0, "5.0": 1, "7.5": 2, "10.0": 3}


def hue_index(h):
    for k, off in STEP.items():
        if h.startswith(k):
            fam = h[len(k):]
            if fam in FAMILIES:
                return FAMILIES.index(fam) * 4 + off
    return None  # e.g. 'N' achromatic


def stress(pred, target):
    pred = np.asarray(pred, float); target = np.asarray(target, float)
    m = np.isfinite(pred) & np.isfinite(target)
    return float(stress_fn(pred[m], target[m]))


def build_pairs(df):
    """Return dict group -> list of (idx_i, idx_j) neighboring pairs (target=1)."""
    df = df.reset_index(drop=True)
    df["hi"] = df["H"].map(hue_index)
    groups = {"varying-H": [], "varying-C": [], "varying-V": []}

    # varying V: fix (H,C), consecutive V (step 1)
    for _, g in df.groupby(["H", "C"]):
        g = g.sort_values("V")
        rows = g[["V"]].to_records(index=True)
        for a, b in zip(rows[:-1], rows[1:]):
            if b["V"] - a["V"] == 1:
                groups["varying-V"].append((a["index"], b["index"]))
    # varying C: fix (H,V), consecutive C (step 2)
    for _, g in df.groupby(["H", "V"]):
        g = g.sort_values("C")
        rows = g[["C"]].to_records(index=True)
        for a, b in zip(rows[:-1], rows[1:]):
            if b["C"] - a["C"] == 2:
                groups["varying-C"].append((a["index"], b["index"]))
    # varying H: fix (C,V), consecutive hue index (step 1 = 2.5 hue apart), C>0 only
    for _, g in df[df["C"] > 0].groupby(["C", "V"]):
        g = g.dropna(subset=["hi"]).sort_values("hi")
        rows = g[["hi"]].to_records(index=True)
        for a, b in zip(rows[:-1], rows[1:]):
            if b["hi"] - a["hi"] == 1:
                groups["varying-H"].append((a["index"], b["index"]))
    return groups


def report(name, coords, groups, dist_fn):
    print(f"\n{'='*64}\n{name}\n{'='*64}")
    all_pred, all_tgt = [], []
    per_group = {}
    for gname, pairs in groups.items():
        I = np.array([p[0] for p in pairs]); J = np.array([p[1] for p in pairs])
        d = dist_fn(coords[I], coords[J])
        t = np.ones(len(d))
        s = stress(d, t)                     # vsl_ial fits optimal k internally
        per_group[gname] = s
        all_pred.append(d); all_tgt.append(t)
        print(f"  {gname:10s}: n={len(d):4d}  STRESS(group-k) = {s:.3f}")
    ap = np.concatenate(all_pred); at = np.concatenate(all_tgt)
    global_s = stress(ap, at)                # single k over all pairs
    group_mean = float(np.mean(list(per_group.values())))
    print(f"  {'-'*40}")
    print(f"  GLOBAL-k STRESS (all {len(ap)} pairs, single k) = {global_s:.3f}")
    print(f"  GROUP-k  STRESS (mean over H/C/V)              = {group_mean:.3f}")
    return global_s, group_mean, per_group


def euclid(a, b):
    return np.linalg.norm(a - b, axis=1)


def main():
    df = pd.read_csv(CSV)
    df["H"] = df["H"].astype(str)
    XYZ = xyY_to_XYZ(df[["x", "y", "Y"]].to_numpy(float) * np.array([1, 1, 1 / 100.0]))
    spaces = {
        # name -> (coords, formula-proper distance)
        "CAM16-LCD (dE, K_L=0.77)": (XYZ_to_CAM16LCD(XYZ), delta_E_CAM16LCD),
        "CAM16-UCS (dE, K_L=1)":    (XYZ_to_CAM16UCS(XYZ), delta_E_CAM16UCS),
        "CAM16-SCD (dE, K_L=1.24)": (XYZ_to_CAM16SCD(XYZ), delta_E_CAM16SCD),
        "CIELAB (dE76)":            (XYZ_to_Lab(XYZ), euclid),
    }
    groups = build_pairs(df)
    print(f"Munsell pairs: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items())
          + f"  (total {sum(len(v) for v in groups.values())})")
    print("Reference (diploma, CAM16-UCS): GLOBAL-k=0.548, GROUP-k=0.354")
    results = {}
    for name, (coords, dist_fn) in spaces.items():
        global_s, group_mean, per_group = report(name, coords, groups, dist_fn)
        results[name] = {"global_k": round(global_s, 4),
                         "group_k_mean": round(group_mean, 4),
                         "per_group": {g: round(v, 4) for g, v in per_group.items()}}
    out = Path("data/analysis"); out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "script": "scripts/cam16_munsell_group_stress.py",
        "date": str(date.today()),
        "protocol": "полный munsell_3-3 (2985 цветов), цепочки H/C/V, цель=1 шаг; "
                    "формульные dE (K_L учтён); STRESS с global-k и group-k",
        "n_pairs": {k: len(v) for k, v in groups.items()},
        "results": results,
    }, open(out / "cam16_munsell.json", "w"), indent=2, ensure_ascii=False)
    print("\nsaved data/analysis/cam16_munsell.json")


if __name__ == "__main__":
    main()
