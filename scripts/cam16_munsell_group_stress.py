"""
CAM16 STRESS on Munsell chains with GLOBAL-k vs GROUP-k (per H/C/V).

Motivation: the learned linear map is evaluated with per-group scaling (Group
STRESS 0.15-0.17), but the CAM16 baseline was reported only with a single global
k (STRESS 0.548). That is not a like-for-like comparison. Here CAM16 gets the same
treatment: separate optimal k and separate STRESS for the varying-H, varying-C and
varying-V chains, plus their mean (Group STRESS).

Target: neighboring colors in a uniform Munsell chain are 1 perceptual step apart.
"""
import numpy as np
import pandas as pd
from colour import xyY_to_XYZ
from colour.models import XYZ_to_CAM16UCS, XYZ_to_CAM16LCD
from colour import XYZ_to_Lab
from vsl_ial.stress import stress as stress_fn

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


def report(name, coords, groups):
    print(f"\n{'='*64}\n{name}\n{'='*64}")
    all_pred, all_tgt = [], []
    per_group = {}
    for gname, pairs in groups.items():
        I = np.array([p[0] for p in pairs]); J = np.array([p[1] for p in pairs])
        d = np.linalg.norm(coords[I] - coords[J], axis=1)
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


def main():
    df = pd.read_csv(CSV)
    df["H"] = df["H"].astype(str)
    XYZ = xyY_to_XYZ(df[["x", "y", "Y"]].to_numpy(float) * np.array([1, 1, 1 / 100.0]))
    coords = {
        "CAM16-LCD": XYZ_to_CAM16LCD(XYZ),
        "CAM16-UCS": XYZ_to_CAM16UCS(XYZ),
        "CIELAB": XYZ_to_Lab(XYZ),
    }
    groups = build_pairs(df)
    print(f"Munsell pairs: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items())
          + f"  (total {sum(len(v) for v in groups.values())})")
    print("Reference (diploma, CAM16-UCS): GLOBAL-k=0.548, GROUP-k=0.354")
    for name in ["CAM16-LCD", "CAM16-UCS", "CIELAB"]:
        report(name, coords[name], groups)


if __name__ == "__main__":
    main()
