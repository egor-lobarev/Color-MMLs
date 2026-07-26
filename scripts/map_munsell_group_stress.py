"""
Symmetric per-group (H/C/V) STRESS: learned linear map vs CAM16, on the SAME
embedding-backed Munsell colors. Object-wise 5-fold CV (split by color center).

Map: one linear A trained on all Munsell chain pairs jointly (target=1 per step),
evaluated per group with per-group optimal k on held-out colors -- the same
group-k treatment given to CAM16 in cam16_munsell_group_stress.py.
"""
import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from colour import xyY_to_XYZ
from colour.models import XYZ_to_CAM16UCS, XYZ_to_CAM16LCD, XYZ_to_CAM16SCD
from colour.difference import (delta_E_CAM16UCS, delta_E_CAM16LCD,
                               delta_E_CAM16SCD)
from sklearn.model_selection import KFold
from vsl_ial.stress import stress as stress_fn

D = Path("data/embeddings/qwen2.5_7B/munsell_colors_describe")
RNG = 42
np.random.seed(RNG); torch.manual_seed(RNG)

FAMILIES = ["R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"]
STEP = {"2.5": 0, "5.0": 1, "7.5": 2, "10.0": 3}


def hue_index(h):
    for k, off in STEP.items():
        if h.startswith(k):
            fam = h[len(k):]
            if fam in FAMILIES:
                return FAMILIES.index(fam) * 4 + off
    return np.nan


def stress(pred, target):
    pred = np.asarray(pred, float); target = np.asarray(target, float)
    m = np.isfinite(pred) & np.isfinite(target)
    return float(stress_fn(pred[m], target[m]))


def load():
    rows, VL, LM = [], [], []
    for d in sorted((p for p in D.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        m = json.load(open(d / "manifest.json"))
        cr = m["csv_row"]
        rows.append({"idx": len(rows), "H": str(cr["H"]), "V": int(cr["V"]),
                     "C": int(cr["C"]), "x": cr["x"], "y": cr["y"], "Y": cr["Y"]})
        VL.append(np.load(d / "vision_pooled_mean.npy").reshape(-1))
        LM.append(np.load(d / "lm_pooled_mean.npy").reshape(-1))
    df = pd.DataFrame(rows)
    df["hi"] = df["H"].map(hue_index)
    return df, np.array(VL, np.float64), np.array(LM, np.float64)


def build_pairs(df):
    g = {"varying-H": [], "varying-C": [], "varying-V": []}
    for _, ch in df.groupby(["H", "C"]):
        ch = ch.sort_values("V"); r = ch[["idx", "V"]].to_records(index=False)
        for a, b in zip(r[:-1], r[1:]):
            if b["V"] - a["V"] == 1: g["varying-V"].append((a["idx"], b["idx"]))
    for _, ch in df.groupby(["H", "V"]):
        ch = ch.sort_values("C"); r = ch[["idx", "C"]].to_records(index=False)
        for a, b in zip(r[:-1], r[1:]):
            if b["C"] - a["C"] == 2: g["varying-C"].append((a["idx"], b["idx"]))
    for _, ch in df[df["C"] > 0].dropna(subset=["hi"]).groupby(["C", "V"]):
        ch = ch.sort_values("hi"); r = ch[["idx", "hi"]].to_records(index=False)
        for a, b in zip(r[:-1], r[1:]):
            if b["hi"] - a["hi"] == 1: g["varying-H"].append((a["idx"], b["idx"]))
    return {k: np.array(v) for k, v in g.items()}


def train_A(X, pairs_all, m, epochs=400, lr=1e-2, reg=1e-3):
    d = X.shape[1]
    I = pairs_all[:, 0]; J = pairs_all[:, 1]
    dx = torch.tensor(X[I] - X[J], dtype=torch.float32)
    y = torch.ones(len(I))
    A = torch.randn(m, d) * (1 / np.sqrt(d)); A.requires_grad_(True)
    opt = torch.optim.Adam([A], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = torch.linalg.norm(dx @ A.T, dim=1)
        (((pred - y) ** 2).mean() + reg * (A ** 2).sum()).backward(); opt.step()
    return A.detach()


def map_dist(A, X, pairs):
    dx = torch.tensor(X[pairs[:, 0]] - X[pairs[:, 1]], dtype=torch.float32)
    return torch.linalg.norm(dx @ A.T, dim=1).numpy()


def main():
    df, VL, LM = load()
    groups = build_pairs(df)
    print(f"colors={len(df)}  pairs: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    # ---- CAM16 baselines on this exact color set (per-group k) ----
    # Formula-proper distances via delta_E_* (K_L is NOT baked into the coords).
    XYZ = xyY_to_XYZ(df[["x", "y", "Y"]].to_numpy(float) * np.array([1, 1, 1 / 100]))
    variants = {
        "CAM16-LCD": (XYZ_to_CAM16LCD(XYZ), delta_E_CAM16LCD),
        "CAM16-UCS": (XYZ_to_CAM16UCS(XYZ), delta_E_CAM16UCS),
        "CAM16-SCD": (XYZ_to_CAM16SCD(XYZ), delta_E_CAM16SCD),
    }
    cam_all = {}
    for vname, (co, dE) in variants.items():
        pg = {}
        for gn, pr in groups.items():
            pg[gn] = stress(dE(co[pr[:, 0]], co[pr[:, 1]]), np.ones(len(pr)))
        cam_all[vname] = pg
        print(f"\n{vname} (this subset, group-k, dE):")
        for gn in groups:
            print(f"  {gn:10s}: {pg[gn]:.3f}")
        print(f"  Group-k mean: {np.mean(list(pg.values())):.3f}")
    cam_pg = cam_all["CAM16-LCD"]  # reference column printed next to the map

    # ---- learned map: object-wise 5-fold, per-group test STRESS ----
    map_res = {}
    for layer, X in [("LM", LM), ("VL", VL)]:
        sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
        map_res[layer] = {}
        for m in [3, 32, 256]:
            fold_pg = {g: [] for g in groups}
            kf = KFold(5, shuffle=True, random_state=RNG)
            for tr, te in kf.split(np.arange(len(df))):
                trs = set(tr.tolist()); tes = set(te.tolist())
                def sub(pr, S): return pr[[a in S and b in S for a, b in pr]]
                train_pairs = np.vstack([sub(groups[g], trs) for g in groups])
                A = train_A(Xs, train_pairs, m)
                for g in groups:
                    tep = sub(groups[g], tes)
                    if len(tep) >= 3:
                        fold_pg[g].append(stress(map_dist(A, Xs, tep), np.ones(len(tep))))
            means = {g: np.mean(v) for g, v in fold_pg.items()}
            gm = np.mean(list(means.values()))
            print(f"\nMAP {layer} m={m:3d} (per-group test STRESS, group-k):")
            for g in groups:
                print(f"  {g:10s}: {means[g]:.3f} ± {np.std(fold_pg[g]):.3f}  (CAM16 {cam_pg[g]:.3f})")
            print(f"  Group-k mean: {gm:.3f}   (CAM16 {np.mean(list(cam_pg.values())):.3f})")
            map_res[layer][m] = {
                "group_k_mean": round(float(gm), 4),
                "per_group": {g: {"mean": round(float(means[g]), 4),
                                  "std": round(float(np.std(fold_pg[g])), 4)}
                              for g in groups}}

    json.dump({
        "script": "scripts/map_munsell_group_stress.py",
        "date": str(date.today()),
        "protocol": "Манселл, подмножество с эмбеддингами (1755 цветов), объектная "
                    "5-фолд CV, MSE-лосс (цель=1 шаг), reg=1e-3, 400 эпох; STRESS "
                    "group-k; CAM16 — формульные dE на том же подмножестве",
        "cam16_on_subset": {v: {g: round(x, 4) for g, x in pg.items()}
                            for v, pg in cam_all.items()},
        "map": map_res,
    }, open(Path("data/analysis") / "map_munsell_groupk.json", "w"),
        indent=2, ensure_ascii=False)
    print("\nsaved data/analysis/map_munsell_groupk.json")


if __name__ == "__main__":
    main()
