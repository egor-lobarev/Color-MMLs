"""
Gram-matrix spectrum of the learned metric vs output dimension m (LM vs VL).

The learned distance is d(x,y)^2 = (x-y)^T G (x-y) with G = A^T A: the map A is
just a factorization of the metric tensor G on embedding space. Eigenvalues of
G (= squared singular values of A) say how many embedding directions the metric
actually uses. Question: why does the LM layer reach optimal quality at m~11-32
while the VL layer keeps improving up to m~3584?

Protocol identical to map_munsell_group_stress.py: Munsell chains (varying-H/C/V,
target=1 per step), object-wise 5-fold CV by color, per-group test STRESS
(group-k mean), epochs=400, lr=1e-2, reg=1e-3, X standardized per-dimension.

For each layer in {LM, VL} and each m in M_LIST, per fold:
  - train A (m x d), eigenvalues lambda_i = sigma_i^2 of G = A^T A,
  - reference spectrum of an UNTRAINED init A (same shape/seed) -- separates
    learned structure from leftover initialization noise,
  - test STRESS,
  - exact SVD-truncation STRESS curve for ALL ranks r=1..m (cheap via projection
    onto right singular vectors) -> functional rank r_q = min r such that
    STRESS(r) <= STRESS(full) + 0.01.

Outputs:
  graphics/fig8_gram_spectrum.(png|pdf)
  data/analysis/gram_spectrum.json  (all numbers)
  data/analysis/gram_spectrum.npz   (spectra + truncation curves)
"""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

RNG = 42
OUT = Path("graphics"); OUT.mkdir(exist_ok=True)
AOUT = Path("data/analysis"); AOUT.mkdir(exist_ok=True, parents=True)
rcParams.update({
    "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 140,
    "font.family": "DejaVu Sans",
})
ACCENT, ACCENT_D, GRAY_D, FLOOR = "#3B6FB5", "#26456f", "#5F6368", "#2E9E6B"

DEV = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mm = load_module("scripts/map_munsell_group_stress.py")


LEEDS_EMB = Path("data/embeddings/qwen2.5_7B/combvd/human_like/leeds")
LEEDS_CSV = Path("data/combvd_pairs.csv")


def load_leeds():
    """Leeds embeddings + pairs (protocol of verify_combvd_leak.py)."""
    ids = sorted(int(d.name) for d in LEEDS_EMB.iterdir() if d.name.isdigit())
    VL, LM = [], []
    remap = {}
    for c in ids:
        remap[c] = len(VL)
        cd = LEEDS_EMB / str(c)
        VL.append(np.load(cd / "vision_pooled_mean.npy").reshape(-1))
        LM.append(np.load(cd / "lm_pooled_mean.npy").reshape(-1))
    I, J, DV = [], [], []
    with open(LEEDS_CSV) as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "leeds":
                continue
            I.append(remap[int(r["i"])]); J.append(remap[int(r["j"])])
            DV.append(float(r["dv"]))
    return (np.array(VL, np.float64), np.array(LM, np.float64),
            np.array(I), np.array(J), np.array(DV, np.float64))


def train_A_group(X, groups_tr, m, seed, epochs=400, lr=1e-2, reg=1e-3):
    """Joint metric learning with the diploma's group-STRESS loss:
    L = mean_g STRESS_g(A) + reg*||A||^2, k_g in closed form inside the graph.
    groups_tr: list of (pairs, targets)."""
    d = X.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    data = []
    for pairs, tgt in groups_tr:
        dx = torch.tensor(X[pairs[:, 0]] - X[pairs[:, 1]], dtype=torch.float32,
                          device=DEV)
        y = torch.tensor(tgt, dtype=torch.float32, device=DEV)
        data.append((dx, y, float((y ** 2).sum())))
    A0 = (torch.randn(m, d, generator=g) * (1 / np.sqrt(d)))
    A = A0.clone().to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([A], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = reg * (A ** 2).sum()
        for dx, y, y2 in data:
            pred = torch.linalg.norm(dx @ A.T, dim=1)
            k = (pred * y).sum() / (pred * pred).sum().clamp_min(1e-12)
            loss = loss + torch.sqrt((((k * pred - y) ** 2).sum() / y2)
                                     .clamp_min(1e-12)) / len(data)
        loss.backward(); opt.step()
    return A.detach().cpu().numpy(), A0.numpy()


def train_A(X, pairs, m, seed, epochs=400, lr=1e-2, reg=1e-3, targets=None):
    """Same as map_munsell_group_stress.train_A but device-aware and seeded.
    targets=None -> unit step (Munsell chains); else measured dv (Leeds)."""
    d = X.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    dx = torch.tensor(X[pairs[:, 0]] - X[pairs[:, 1]], dtype=torch.float32,
                      device=DEV)
    y = (torch.ones(len(pairs), device=DEV) if targets is None
         else torch.tensor(targets, dtype=torch.float32, device=DEV))
    A0 = (torch.randn(m, d, generator=g) * (1 / np.sqrt(d)))
    A = A0.clone().to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([A], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = torch.linalg.norm(dx @ A.T, dim=1)
        (((pred - y) ** 2).mean() + reg * (A ** 2).sum()).backward()
        opt.step()
    return A.detach().cpu().numpy(), A0.numpy()


def truncation_stress_curve(A, Xs, test_groups):
    """Exact group-k mean test STRESS for EVERY truncation rank r=1..m.

    ||A_r dx|| = ||diag(S_1..r) (Vt dx)_1..r||  -> cumulative sums over the
    singular-vector index give all ranks at once. test_groups: list of pair
    arrays (target=1) or of (pairs, targets) tuples.

    STRESS with per-group optimal k: stress^2 = 1 - (sum d*t)^2/(sum d^2 * sum t^2).
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    per_group = []
    for grp in test_groups:
        pairs, tgt = grp if isinstance(grp, tuple) else (grp, np.ones(len(grp)))
        dx = Xs[pairs[:, 0]] - Xs[pairs[:, 1]]           # (n, d)
        proj = (dx @ Vt.T) * S                            # (n, m)
        cum = np.cumsum(proj ** 2, axis=1)                # (n, m) dist^2 at rank r
        dists = np.sqrt(cum)                              # (n, m)
        num = (dists * tgt[:, None]).sum(0)               # (m,) sum d*t
        den = cum.sum(0) * (tgt ** 2).sum()               # (m,) sum d^2 * sum t^2
        st = np.sqrt(np.maximum(0.0, 1.0 - num ** 2 / den))
        per_group.append(st)
    return S ** 2, np.mean(per_group, axis=0)             # eigenvalues, curve(r)


def effective_ranks(lam):
    lam = np.sort(np.asarray(lam))[::-1]
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    n95 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)
    return pr, n95


def functional_rank(curve, tol=0.01):
    full = curve[-1]
    return int(np.argmax(curve <= full + tol) + 1)


M_LIST_FULL = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3584]


def make_figure(summary, spectra, init_spectra, M_LIST,
                fname="fig8_gram_spectrum",
                suptitle=("Матрица Грама G = AᵀA обученной метрики: языковой декодер "
                          "держит компактное цветовое ядро, визуальный энкодер "
                          "распределяет метрику по сотням направлений"),
                ref_line=(0.288, "CAM16-LCD (0.288)"),
                ylabel3="STRESS на тесте (group-k)",
                ylim_spec=(1e-3, 8), annotate="noise"):
    def corner(ax, txt):
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold", color="#333")

    show_m = [32, 256, 1024, 3584]
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.8))
    a1, a2, a3, a4 = axes
    cmap = plt.get_cmap("Blues")
    shades = [cmap(0.35 + 0.6 * i / (len(show_m) - 1)) for i in range(len(show_m))]

    for ax, layer, ttl in [(a1, "LM", "а)  LM"), (a2, "VL", "б)  VL")]:
        for c, m in zip(shades, show_m):
            lam = spectra[(layer, m)]
            ax.plot(np.arange(1, len(lam) + 1), lam / lam[0], color=c, lw=2.0,
                    label=f"m={m}")
        lam0 = init_spectra[(layer, 3584)]
        lam = spectra[(layer, 3584)]
        ax.plot(np.arange(1, len(lam0) + 1), lam0 / lam[0], color="#999999",
                lw=1.3, ls="--", label="A до обучения (шум)")
        if annotate == "noise":
            n_above = int((lam > lam0[0]).sum())
            ax.annotate(f"{n_above} λ над полкой шума",
                        xy=(max(n_above, 1), lam[max(n_above - 1, 0)] / lam[0]),
                        xytext=(max(n_above, 1) * 6, 2.6),
                        arrowprops=dict(arrowstyle="->", color="#666", lw=0.9),
                        fontsize=12, color="#333")
        elif annotate == "rq":
            rq = int(round(summary[layer]["3584"]["rq_mean"]))
            ax.annotate(f"r_q ≈ {rq}: дальше метрика пуста",
                        xy=(rq, lam[rq - 1] / lam[0]), xytext=(rq * 6, 1.2),
                        arrowprops=dict(arrowstyle="->", color="#666", lw=0.9),
                        fontsize=12, color="#333")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(*ylim_spec)
        ax.set_xlabel("Номер собственного значения (log)")
        ax.set_ylabel("λᵢ / λ₁ (log)")
        corner(ax, ttl)
        ax.legend(frameon=False, fontsize=11, loc="lower left")
        ax.grid(color="#EEE")

    ms = np.array(M_LIST)
    for layer, col in [("LM", ACCENT), ("VL", ACCENT_D)]:
        st = [summary[layer][str(m)]["stress_mean"] for m in M_LIST]
        er = [summary[layer][str(m)]["stress_std"] for m in M_LIST]
        a3.errorbar(ms, st, yerr=er, fmt="o-", color=col, lw=2.2, ms=5,
                    capsize=2, label=layer)
    a3.axhline(ref_line[0], color=GRAY_D, ls="--", lw=1.2)
    a3.text(ms[-1], ref_line[0] + 0.006, ref_line[1], ha="right", color=GRAY_D,
            fontsize=12)
    a3.set_xscale("log")
    a3.set_xlabel("Выходная размерность m (log)")
    a3.set_ylabel(ylabel3)
    corner(a3, "в)")
    a3.legend(frameon=False, fontsize=13); a3.grid(color="#EEE")

    for layer, col in [("LM", ACCENT), ("VL", ACCENT_D)]:
        rq = [np.mean(summary[layer][str(m)]["rq_all"]) for m in M_LIST]
        er = [np.std(summary[layer][str(m)]["rq_all"]) for m in M_LIST]
        a4.errorbar(ms, rq, yerr=er, fmt="o-", color=col, lw=2.2, ms=5,
                    capsize=2, label=layer)
    a4.plot(ms, ms, color="#BBB", lw=1, ls=":")
    a4.text(ms[2], ms[2] * 1.5, "r = m", color="#999", fontsize=11, rotation=38)
    a4.set_xscale("log"); a4.set_yscale("log")
    a4.set_xlabel("Выходная размерность m (log)")
    a4.set_ylabel("Функциональный ранг r_q")
    corner(a4, "г)")
    a4.legend(frameon=False, fontsize=13); a4.grid(color="#EEE")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")


JOINT_FIG_KW = dict(
    fname="fig9_gram_spectrum_joint",
    suptitle=("Совместная метрика (Манселл + Leeds, групповой STRESS-лосс): оба масштаба "
              "различий умещаются в общее ядро ~20 направлений, асимметрия LM/VL исчезает"),
    # CAM16-LCD на тех же 4 группах: varying-V 0.074 / C 0.286 / H 0.504 (подмножество
    # 1755 цветов, map_munsell_group_stress.py) + Leeds 0.322 (verify_combvd_leak.py)
    ref_line=(0.297, "CAM16-LCD, те же 4 группы (0.297)"),
    ylabel3="Group STRESS на тесте (4 группы)",
    # reg=0 -> масштаб A свободно растёт, полка инициализации уходит на ~4 порядка
    # вниз; смысловую границу задаёт функциональный ранг, его и аннотируем
    ylim_spec=(1e-5, 8), annotate="rq")


def replot(joint=False):
    tag = "_joint" if joint else ""
    summary = json.load(open(AOUT / f"gram_spectrum{tag}.json"))
    z = np.load(AOUT / f"gram_spectrum{tag}.npz")
    spectra = {}; init_spectra = {}
    for key in z.files:
        parts = key.split("_")
        if parts[0] == "lam":
            spectra[(parts[1], int(parts[2]))] = z[key]
        elif parts[0] == "lam0":
            init_spectra[(parts[1], int(parts[2]))] = z[key]
    make_figure(summary, spectra, init_spectra, M_LIST_FULL,
                **(JOINT_FIG_KW if joint else {}))


def run_joint(pilot=False):
    """Joint Munsell+Leeds training (diploma protocol: 4-group STRESS loss),
    center-wise 5-fold CV in BOTH datasets, Gram-spectrum analysis per m."""
    np.random.seed(RNG); torch.manual_seed(RNG)
    df, VLm, LMm = mm.load()
    mgroups = mm.build_pairs(df)
    VLl, LMl, I, J, DV = load_leeds()
    nm, nl = len(df), len(VLl)
    leeds_pairs = np.stack([I + nm, J + nm], 1)
    print(f"device={DEV} munsell={nm} leeds_colors={nl} leeds_pairs={len(I)}")

    M_LIST = M_LIST_FULL
    FOLDS = 5
    if pilot:
        M_LIST = [32, 3584]; FOLDS = 1

    from sklearn.model_selection import KFold
    kf = KFold(5, shuffle=True, random_state=RNG)
    splits = [(np.concatenate([mtr, ltr + nm]), np.concatenate([mte, lte + nm]))
              for (mtr, mte), (ltr, lte)
              in zip(kf.split(np.arange(nm)), kf.split(np.arange(nl)))][:FOLDS]

    def sub(pr, S):
        return pr[[a in S and b in S for a, b in pr]]

    GN = list(mgroups) + ["leeds"]

    def groups_for(S):
        out = []
        for g in mgroups:
            p = sub(mgroups[g], S)
            out.append((p, np.ones(len(p))))
        mask = np.array([a in S and b in S for a, b in leeds_pairs])
        out.append((leeds_pairs[mask], DV[mask]))
        return out

    res, spectra, init_spectra, curves = {}, {}, {}, {}
    for layer, X in [("LM", np.vstack([LMm, LMl])),
                     ("VL", np.vstack([VLm, VLl]))]:
        # per-dimension std over the combined color set (scale absorbed by A)
        sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
        res[layer] = {}
        for m in M_LIST:
            r = dict(stress=[], pr=[], n95=[], rq=[])
            gstress = {g: [] for g in GN}
            lam_f, lam0_f, cur_f = [], [], []
            for fi, (tr, te) in enumerate(splits):
                trs, tes = set(tr.tolist()), set(te.tolist())
                t0 = time.time()
                # reg=0: the group-STRESS loss is scale-invariant, so ANY L2
                # (even 1e-5*sum A^2) dominates Adam updates at large m and
                # blocks fitting (train STRESS stuck at 0.2-0.4 vs 0.04 without).
                # Likely why the diploma saw tiny m_opt (14/32) for joint runs.
                A, A0 = train_A_group(Xs, groups_for(trs), m,
                                      seed=RNG + 1000 * fi + m, reg=0.0)
                test_groups = [g for g in groups_for(tes) if len(g[0]) >= 3]
                lam, curve = truncation_stress_curve(A, Xs, test_groups)
                for g, (p, tgt) in zip(GN, groups_for(tes)):
                    if len(p) >= 3:
                        pred = np.linalg.norm((Xs[p[:, 0]] - Xs[p[:, 1]]) @ A.T,
                                              axis=1)
                        gstress[g].append(mm.stress(pred, tgt))
                lam0 = np.linalg.svd(A0, compute_uv=False) ** 2
                pr, n95 = effective_ranks(lam)
                rq = functional_rank(curve)
                r["stress"].append(float(curve[-1]))
                r["pr"].append(pr); r["n95"].append(n95); r["rq"].append(rq)
                lam_f.append(np.sort(lam)[::-1])
                lam0_f.append(np.sort(lam0)[::-1])
                cur_f.append(curve)
                print(f"[joint {layer}] m={m:4d} fold{fi} "
                      f"GroupSTRESS={curve[-1]:.3f} r_q={rq} "
                      f"leeds={gstress['leeds'][-1]:.3f} ({time.time()-t0:.0f}s)")
            res[layer][m] = {k: [float(x) for x in v] for k, v in r.items()}
            res[layer][m]["groups"] = {g: [float(x) for x in v]
                                       for g, v in gstress.items()}
            spectra[(layer, m)] = np.mean(lam_f, axis=0)
            init_spectra[(layer, m)] = np.mean(lam0_f, axis=0)
            curves[(layer, m)] = np.mean(cur_f, axis=0)

    summary = {layer: {str(m): {
        "stress_mean": float(np.mean(v["stress"])),
        "stress_std": float(np.std(v["stress"])),
        "pr_mean": float(np.mean(v["pr"])),
        "n95_mean": float(np.mean(v["n95"])),
        "rq_mean": float(np.mean(v["rq"])),
        "rq_all": v["rq"],
        "groups": {g: {"mean": float(np.mean(s)), "std": float(np.std(s))}
                   for g, s in v["groups"].items()},
    } for m, v in res[layer].items()} for layer in res}
    json.dump(summary, open(AOUT / "gram_spectrum_joint.json", "w"), indent=2)
    np.savez(AOUT / "gram_spectrum_joint.npz",
             **{f"lam_{l}_{m}": v for (l, m), v in spectra.items()},
             **{f"lam0_{l}_{m}": v for (l, m), v in init_spectra.items()},
             **{f"curve_{l}_{m}": v for (l, m), v in curves.items()})
    print(json.dumps(summary, indent=2))
    if pilot:
        return
    make_figure(summary, spectra, init_spectra, M_LIST, **JOINT_FIG_KW)


def run_cross(m=256):
    """Cross-dataset generalization for the article's Table 2: train on the FULL
    source dataset, test on the FULL target (no split needed -- disjoint data)."""
    np.random.seed(RNG); torch.manual_seed(RNG)
    df, VLm, LMm = mm.load()
    mgroups = mm.build_pairs(df)
    VLl, LMl, I, J, DV = load_leeds()
    leeds_pairs = np.stack([I, J], 1)
    all_mun_pairs = np.vstack([mgroups[g] for g in mgroups])
    out = {}
    for lname, Xm, Xl in [("LM", LMm, LMl), ("VL", VLm, VLl)]:
        sd = Xm.std(0); sd[sd == 0] = 1
        A, _ = train_A(Xm / sd, all_mun_pairs, m, seed=RNG)
        dxl = (Xl / sd)[leeds_pairs[:, 0]] - (Xl / sd)[leeds_pairs[:, 1]]
        m2l = mm.stress(np.linalg.norm(dxl @ A.T, axis=1), DV)
        sd2 = Xl.std(0); sd2[sd2 == 0] = 1
        A2, _ = train_A(Xl / sd2, leeds_pairs, m, seed=RNG, targets=DV,
                        epochs=350)
        pg = {}
        for g in mgroups:
            p = mgroups[g]
            dxm = (Xm / sd2)[p[:, 0]] - (Xm / sd2)[p[:, 1]]
            pg[g] = mm.stress(np.linalg.norm(dxm @ A2.T, axis=1),
                              np.ones(len(p)))
        out[lname] = {"munsell_to_leeds": float(m2l),
                      "leeds_to_munsell_groupk": float(np.mean(list(pg.values()))),
                      "leeds_to_munsell_groups": {k: float(v) for k, v in pg.items()}}
        print(f"[{lname}] munsell->leeds {m2l:.3f}   leeds->munsell "
              f"{np.mean(list(pg.values())):.3f} "
              + " ".join(f"{g}={v:.3f}" for g, v in pg.items()))
    json.dump(out, open(AOUT / "cross_dataset.json", "w"), indent=2)
    print("saved data/analysis/cross_dataset.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--joint", action="store_true",
                    help="joint Munsell+Leeds training (diploma 4-group protocol)")
    ap.add_argument("--cross", action="store_true",
                    help="cross-dataset eval (train full Munsell -> test Leeds and v.v.)")
    ap.add_argument("--replot", action="store_true",
                    help="redraw figure from saved json/npz, no retraining")
    args = ap.parse_args()
    if args.replot:
        replot(joint=args.joint); return
    if args.cross:
        run_cross(); return
    if args.joint:
        run_joint(pilot=args.pilot); return

    np.random.seed(RNG); torch.manual_seed(RNG)
    df, VL, LM = mm.load()
    groups = mm.build_pairs(df)
    print(f"device={DEV} colors={len(df)} pairs: "
          + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    M_LIST = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3584]
    FOLDS = 5
    if args.pilot:
        M_LIST = [32, 3584]; FOLDS = 1

    from sklearn.model_selection import KFold
    kf = KFold(5, shuffle=True, random_state=RNG)
    splits = list(kf.split(np.arange(len(df))))[:FOLDS]

    def sub(pr, S):
        return pr[[a in S and b in S for a, b in pr]]

    res = {}      # res[layer][m] = dict(stress=[...], pr=[...], n95=[...], rq=[...])
    spectra = {}  # spectra[(layer, m)] = mean sorted eigenvalues over folds
    init_spectra = {}
    curves = {}   # truncation curves (mean over folds)
    for layer, X in [("LM", LM), ("VL", VL)]:
        sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
        res[layer] = {}
        for m in M_LIST:
            r = dict(stress=[], pr=[], n95=[], rq=[])
            lam_f, lam0_f, cur_f = [], [], []
            for fi, (tr, te) in enumerate(splits):
                trs, tes = set(tr.tolist()), set(te.tolist())
                train_pairs = np.vstack([sub(groups[g], trs) for g in groups])
                test_groups = [sub(groups[g], tes) for g in groups]
                t0 = time.time()
                A, A0 = train_A(Xs, train_pairs, m,
                                seed=RNG + 1000 * fi + m)
                lam, curve = truncation_stress_curve(A, Xs, test_groups)
                lam0 = np.linalg.svd(A0, compute_uv=False) ** 2
                pr, n95 = effective_ranks(lam)
                rq = functional_rank(curve)
                r["stress"].append(float(curve[-1]))
                r["pr"].append(pr); r["n95"].append(n95); r["rq"].append(rq)
                lam_f.append(np.sort(lam)[::-1])
                lam0_f.append(np.sort(lam0)[::-1])
                cur_f.append(curve)
                print(f"[{layer}] m={m:4d} fold{fi} "
                      f"STRESS={curve[-1]:.3f} PR={pr:.1f} n95={n95} "
                      f"r_q={rq} ({time.time()-t0:.0f}s)")
            res[layer][m] = {k: [float(x) for x in v] for k, v in r.items()}
            spectra[(layer, m)] = np.mean(lam_f, axis=0)
            init_spectra[(layer, m)] = np.mean(lam0_f, axis=0)
            curves[(layer, m)] = np.mean(cur_f, axis=0)

    # ---------------- save numbers ----------------
    summary = {layer: {str(m): {
        "stress_mean": float(np.mean(v["stress"])),
        "stress_std": float(np.std(v["stress"])),
        "pr_mean": float(np.mean(v["pr"])),
        "n95_mean": float(np.mean(v["n95"])),
        "rq_mean": float(np.mean(v["rq"])),
        "rq_all": v["rq"],
    } for m, v in res[layer].items()} for layer in res}
    json.dump(summary, open(AOUT / "gram_spectrum.json", "w"), indent=2)
    np.savez(AOUT / "gram_spectrum.npz",
             **{f"lam_{l}_{m}": v for (l, m), v in spectra.items()},
             **{f"lam0_{l}_{m}": v for (l, m), v in init_spectra.items()},
             **{f"curve_{l}_{m}": v for (l, m), v in curves.items()})
    print(json.dumps(summary, indent=2))
    if args.pilot:
        return

    make_figure(summary, spectra, init_spectra, M_LIST)


if __name__ == "__main__":
    main()
