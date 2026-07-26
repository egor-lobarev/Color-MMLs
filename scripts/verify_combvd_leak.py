"""
Verify the COMBVD-Leeds data leak and re-check reported numbers.

Task 1 of the publication-2 plan: the diploma split COMBVD *by pair*, so a color
center can appear in both train and test (leakage). Here we re-run metric learning
under two protocols and compare:

    (A) split BY PAIR      -> reproduces the (leaky) diploma setup
    (B) split BY CENTER    -> correct: no color center in both train & test

We also recompute the CAM16 colorimetric baseline in BOTH variants (UCS and LCD),
since the final metric adopted for the paper is CAM16-LCD.

Data on disk: Qwen2.5-VL-7B, prompt = "human_like", subset = leeds only.
"""
import json
import csv
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from colour import xyY_to_XYZ
from colour.models import XYZ_to_CAM16UCS, XYZ_to_CAM16LCD, XYZ_to_CAM16SCD
from colour.difference import (delta_E_CAM16UCS, delta_E_CAM16LCD,
                               delta_E_CAM16SCD)
from colour import XYZ_to_Lab, delta_E

from vsl_ial.stress import stress as stress_fn

RNG = 42
np.random.seed(RNG)
torch.manual_seed(RNG)

EMB_DIR = Path("data/embeddings/qwen2.5_7B/combvd/human_like/leeds")
PAIRS_CSV = Path("data/combvd_pairs.csv")
NOISE_FLOOR = 0.0896  # STRESS of repeated human measurements (diploma)


# ----------------------------------------------------------------------------- data
def load_leeds():
    centers = sorted(int(d.name) for d in EMB_DIR.iterdir() if d.name.isdigit())
    n = max(centers) + 1
    d = 3584
    VL = np.full((n, d), np.nan, np.float64)
    LM = np.full((n, d), np.nan, np.float64)
    XYZ = np.full((n, 3), np.nan, np.float64)
    for c in centers:
        cd = EMB_DIR / str(c)
        VL[c] = np.load(cd / "vision_pooled_mean.npy").reshape(-1)
        LM[c] = np.load(cd / "lm_pooled_mean.npy").reshape(-1)
        m = json.load(open(cd / "manifest.json"))
        XYZ[c] = xyY_to_XYZ(np.array(m["xyY"], float))
    return VL, LM, XYZ, np.array(centers)


def load_pairs():
    I, J, DV = [], [], []
    with open(PAIRS_CSV) as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "leeds":
                continue
            I.append(int(r["i"])); J.append(int(r["j"])); DV.append(float(r["dv"]))
    return np.array(I), np.array(J), np.array(DV)


def stress(d_true, d_pred):
    d_true = np.asarray(d_true, float); d_pred = np.asarray(d_pred, float)
    m = np.isfinite(d_true) & np.isfinite(d_pred)
    return float(stress_fn(d_pred[m], d_true[m]))


# --------------------------------------------------------------------- baselines
def cam_baselines(XYZ, I, J, DV):
    """Formula-proper distances. NB: colour-science does NOT bake K_L into the
    UCS/LCD/SCD coordinates, so for K_L != 1 (LCD 0.77, SCD 1.24) the distance
    must go through delta_E_*; Euclid on coords is only correct for UCS."""
    out = {}
    for name, fn, dE in [("CAM16-UCS (dE)", XYZ_to_CAM16UCS, delta_E_CAM16UCS),
                         ("CAM16-LCD (dE)", XYZ_to_CAM16LCD, delta_E_CAM16LCD),
                         ("CAM16-SCD (dE)", XYZ_to_CAM16SCD, delta_E_CAM16SCD)]:
        try:
            coords = fn(XYZ)
            out[name] = stress(DV, dE(coords[I], coords[J]))
        except Exception as e:
            out[name] = f"err:{e}"
    # raw CAM16 correlates, no UCS compression: Euclid on (J, M cos h, M sin h)
    try:
        from colour.appearance import XYZ_to_CAM16, VIEWING_CONDITIONS_CAM16
        from colour.models import xy_to_XYZ
        XYZ_w = xy_to_XYZ(np.array([0.3127, 0.3290])) * 100
        spec = XYZ_to_CAM16(XYZ * 100, XYZ_w, L_A=64 / np.pi * 0.2 * 100, Y_b=20,
                            surround=VIEWING_CONDITIONS_CAM16["Average"])
        raw = np.stack([spec.J, spec.M * np.cos(np.radians(spec.h)),
                        spec.M * np.sin(np.radians(spec.h))], axis=-1)
        out["raw CAM16 (J,M,h Euclid)"] = stress(
            DV, np.linalg.norm(raw[I] - raw[J], axis=1))
    except Exception as e:
        out["raw CAM16 (J,M,h Euclid)"] = f"err:{e}"
    # CIELAB / CIEDE2000 anchors (D65 white)
    Lab = XYZ_to_Lab(XYZ)
    d76 = np.linalg.norm(Lab[I] - Lab[J], axis=1)
    out["CIELAB dE76"] = stress(DV, d76)
    try:
        d00 = delta_E(Lab[I], Lab[J], method="CIE 2000")
        out["CIEDE2000"] = stress(DV, d00)
    except Exception as e:
        out["CIEDE2000"] = f"err:{e}"
    return out


# --------------------------------------------------------------- metric learning
def train_A(X, I, J, DV, m, epochs=350, lr=1e-2, reg=1e-3, device="cpu"):
    """Learn A (m x d): min mean( (||A (x_i - x_j)|| - dv)^2 ) + reg||A||^2."""
    d = X.shape[1]
    dx = torch.tensor(X[I] - X[J], dtype=torch.float32, device=device)
    y = torch.tensor(DV, dtype=torch.float32, device=device)
    A = torch.randn(m, d, device=device) * (1.0 / np.sqrt(d))
    A.requires_grad_(True)
    opt = torch.optim.Adam([A], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = torch.linalg.norm(dx @ A.T, dim=1)
        loss = ((pred - y) ** 2).mean() + reg * (A ** 2).sum()
        loss.backward(); opt.step()
    return A.detach()


def eval_A(A, X, I, J, DV):
    dx = torch.tensor(X[I] - X[J], dtype=torch.float32)
    pred = torch.linalg.norm(dx @ A.T, dim=1).numpy()
    return stress(DV, pred)


# feature scaling helps optimization; differences are mean-invariant, std scaling
# is absorbed by A, so it does not change the attainable STRESS optimum.
def scale_feats(X, train_centers):
    sd = X[train_centers].std(0)
    sd[sd == 0] = 1.0
    return X / sd


def run_protocol(name, X, I, J, DV, centers, by_center, ms, n_splits=5):
    print(f"\n{'='*74}\n{name}\n{'='*74}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    results = {m: [] for m in ms}
    if by_center:
        split_units = centers
    else:
        split_units = np.arange(len(I))
    for tr, te in kf.split(split_units):
        if by_center:
            tr_c = set(centers[tr]); te_c = set(centers[te])
            tr_mask = np.array([i in tr_c and j in tr_c for i, j in zip(I, J)])
            te_mask = np.array([i in te_c and j in te_c for i, j in zip(I, J)])
            train_centers = centers[tr]
        else:
            tr_mask = np.zeros(len(I), bool); tr_mask[tr] = True
            te_mask = np.zeros(len(I), bool); te_mask[te] = True
            train_centers = centers  # scaling on all (mild; features only)
        Xs = scale_feats(X, train_centers)
        for m in ms:
            A = train_A(Xs, I[tr_mask], J[tr_mask], DV[tr_mask], m)
            s = eval_A(A, Xs, I[te_mask], J[te_mask], DV[te_mask])
            results[m].append(s)
        # report fold-0 pair counts once
        if len(results[ms[0]]) == 1:
            print(f"  fold sizes: train_pairs={tr_mask.sum()} test_pairs={te_mask.sum()}"
                  + (f" (straddling dropped={len(I)-tr_mask.sum()-te_mask.sum()})" if by_center else ""))
    for m in ms:
        arr = np.array(results[m])
        print(f"  m={m:5d}  STRESS = {arr.mean():.3f} ± {arr.std():.3f}")
    return results


def main():
    VL, LM, XYZ, centers = load_leeds()
    I, J, DV = load_pairs()
    print(f"Leeds: {len(centers)} centers, {len(I)} pairs, d=3584, prompt=human_like")

    print("\n" + "="*74 + "\nCAM16 / CIE baselines (STRESS vs human dv) -- for scale calibration\n" + "="*74)
    baselines = cam_baselines(XYZ, I, J, DV)
    for k, v in baselines.items():
        print(f"  {k:24s}: {v if isinstance(v,str) else f'{v:.3f}'}")
    print(f"  {'noise floor (repeats)':24s}: {NOISE_FLOOR:.3f}")
    print("  [diploma reported CAM16-UCS on Leeds = 0.271]")

    ms = [3, 32, 256, 1024]
    proto = {}
    for layer, X in [("LM", LM), ("VL", VL)]:
        print(f"\n########## LAYER {layer} ##########")
        ra = run_protocol(f"(A) SPLIT BY PAIR  [leaky, reproduces diploma] -- {layer}",
                          X, I, J, DV, centers, by_center=False, ms=ms)
        rb = run_protocol(f"(B) SPLIT BY CENTER [correct] -- {layer}",
                          X, I, J, DV, centers, by_center=True, ms=ms)
        proto[layer] = {
            "by_pair_leaky": {m: {"mean": round(float(np.mean(v)), 4),
                                  "std": round(float(np.std(v)), 4)}
                              for m, v in ra.items()},
            "by_center": {m: {"mean": round(float(np.mean(v)), 4),
                              "std": round(float(np.std(v)), 4)}
                          for m, v in rb.items()},
        }

    out = Path("data/analysis"); out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "script": "scripts/verify_combvd_leak.py",
        "date": str(date.today()),
        "protocol": "COMBVD-Leeds (307 пар), Qwen-7B prompt=human_like; MSE-лосс на "
                    "dv, reg=1e-3, 350 эпох; (A) сплит по парам [утечка], (B) по "
                    "цветовым центрам [корректно], 5-фолд CV; бейзлайны — формульные dE",
        "baselines": {k: (round(v, 4) if not isinstance(v, str) else v)
                      for k, v in baselines.items()},
        "noise_floor_repeats": NOISE_FLOOR,
        "map": proto,
    }, open(out / "verify_combvd_leak.json", "w"), indent=2, ensure_ascii=False)
    print("\nsaved data/analysis/verify_combvd_leak.json")


if __name__ == "__main__":
    main()
