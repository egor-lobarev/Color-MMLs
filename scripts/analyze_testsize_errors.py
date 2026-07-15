"""
(1) Dependence of test STRESS on the test-set size (data efficiency / stability):
    Munsell (object-wise split) and COMBVD-Leeds (center-wise split), m=256.
(2) Error analysis across the Munsell body (Value x Chroma), two variants:
    - fig7_error_heatmap    : metric-map step error |k_g*d_hat - 1| on held-out
      pairs, unitless (fraction of one Munsell step; per-pair STRESS contribution).
    - fig7_error_heatmap_dE : LIM26-Fig.3 analog in familiar dE units — linear
      decoding of CAM16-LCD coordinates from embeddings, per-test-color ||y_hat-y||.
      (A Munsell step has no fixed dE equivalent, so the dE view uses coordinate
      decoding instead of the metric map.)

Run `python scripts/analyze_testsize_errors.py --heatmap-only` to skip the sweep.

Produces (graphics/):
  fig6_testsize.(png|pdf)
  fig7_error_heatmap.(png|pdf), fig7_error_heatmap_dE.(png|pdf)
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import KFold

RNG = 42
np.random.seed(RNG); torch.manual_seed(RNG)
OUT = Path("graphics"); OUT.mkdir(exist_ok=True)
rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 140,
    "font.family": "DejaVu Sans",
})
ACCENT, ACCENT_D, GRAY_D = "#3B6FB5", "#26456f", "#5F6368"


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mm = load_module("scripts/map_munsell_group_stress.py")   # Munsell loaders/train
vc = load_module("scripts/verify_combvd_leak.py")         # Leeds loaders/train

FRACS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]   # test-set fraction
SEEDS = [0, 1, 2]
M = 256


def sub(pr, S):
    return pr[[a in S and b in S for a, b in pr]]


# ------------------------------------------------- 1a) Munsell test-size sweep
def munsell_sweep(df, groups, layers):
    res = {name: {f: [] for f in FRACS} for name in layers}
    n = len(df)
    for name, X in layers.items():
        sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
        for f in FRACS:
            for seed in SEEDS:
                idx = np.random.RandomState(seed).permutation(n)
                cut = int((1 - f) * n)
                tr, te = set(idx[:cut].tolist()), set(idx[cut:].tolist())
                train_pairs = np.vstack([sub(groups[g], tr) for g in groups])
                if len(train_pairs) < 50:
                    res[name][f].append(np.nan); continue
                A = mm.train_A(Xs, train_pairs, m=M)
                pg = [mm.stress(mm.map_dist(A, Xs, sub(groups[g], te)),
                                np.ones(len(sub(groups[g], te))))
                      for g in groups if len(sub(groups[g], te)) >= 3]
                res[name][f].append(np.mean(pg))
            print(f"  munsell {name} f={f}: "
                  f"{np.nanmean(res[name][f]):.3f}±{np.nanstd(res[name][f]):.3f}")
    return res


# ------------------------------------------------- 1b) Leeds test-size sweep
def leeds_sweep():
    VLe, LMe, XYZ, centers = vc.load_leeds()
    I, J, DV = vc.load_pairs()
    res = {name: {f: [] for f in FRACS} for name in ("VL", "LM")}
    for name, X in [("VL", VLe), ("LM", LMe)]:
        for f in FRACS:
            for seed in SEEDS:
                idx = np.random.RandomState(seed).permutation(len(centers))
                cut = int((1 - f) * len(centers))
                trc = set(centers[idx[:cut]].tolist())
                tec = set(centers[idx[cut:]].tolist())
                trm = np.array([a in trc and b in trc for a, b in zip(I, J)])
                tem = np.array([a in tec and b in tec for a, b in zip(I, J)])
                if trm.sum() < 30 or tem.sum() < 10:
                    res[name][f].append(np.nan); continue
                Xs = vc.scale_feats(X, centers[idx[:cut]])
                A = vc.train_A(Xs, I[trm], J[trm], DV[trm], M)
                res[name][f].append(vc.eval_A(A, Xs, I[tem], J[tem], DV[tem]))
            print(f"  leeds {name} f={f}: "
                  f"{np.nanmean(res[name][f]):.3f}±{np.nanstd(res[name][f]):.3f}")
    return res


# ------------------------------------------------- 2) error heatmap (Munsell)
def error_heatmap(df, groups, layers):
    err_maps = {}
    for name, X in layers.items():
        sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
        color_err = {i: [] for i in range(len(df))}
        kf = KFold(5, shuffle=True, random_state=RNG)
        for tr_i, te_i in kf.split(np.arange(len(df))):
            tr, te = set(tr_i.tolist()), set(te_i.tolist())
            train_pairs = np.vstack([sub(groups[g], tr) for g in groups])
            A = mm.train_A(Xs, train_pairs, m=M)
            for g in groups:
                tep = sub(groups[g], te)
                if len(tep) < 3:
                    continue
                d = mm.map_dist(A, Xs, tep)
                k = float(np.dot(d, np.ones(len(d))) / np.dot(d, d))
                e = np.abs(k * d - 1.0)
                for (a, b), ei in zip(tep, e):
                    color_err[a].append(ei); color_err[b].append(ei)
        rows = [{"V": df.iloc[i]["V"], "C": df.iloc[i]["C"],
                 "err": np.mean(v)} for i, v in color_err.items() if v]
        piv = pd.DataFrame(rows).groupby(["V", "C"])["err"].mean().unstack()
        err_maps[name] = piv
        print(f"  heatmap {name}: mean={np.nanmean(piv.values):.3f}, "
              f"max={np.nanmax(piv.values):.3f}")
    return err_maps


# ------------------------------------- 2b) error heatmap in dE units (LIM26-style)
def error_heatmap_dE(df, layers):
    """Linear decoding of CAM16-LCD coords (Ridge, object-wise 5-fold);
    per-test-color error ||y_hat - y|| in dE units of CAM16-LCD."""
    from colour import xyY_to_XYZ
    from colour.models import XYZ_to_CAM16LCD
    from sklearn.linear_model import Ridge

    XYZ = xyY_to_XYZ(df[["x", "y", "Y"]].to_numpy(float) * np.array([1, 1, 1 / 100]))
    Y = XYZ_to_CAM16LCD(XYZ)
    maps = {}
    for name, X in layers.items():
        sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
        err = np.full(len(df), np.nan)
        for tr_i, te_i in KFold(5, shuffle=True, random_state=RNG).split(Xs):
            reg = Ridge(alpha=1.0).fit(Xs[tr_i], Y[tr_i])
            err[te_i] = np.linalg.norm(reg.predict(Xs[te_i]) - Y[te_i], axis=1)
        piv = (pd.DataFrame({"V": df["V"], "C": df["C"], "err": err})
               .groupby(["V", "C"])["err"].mean().unstack())
        maps[name] = piv
        print(f"  dE heatmap {name}: mean={np.nanmean(err):.3f} dE, "
              f"max cell={np.nanmax(piv.values):.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    vmax = max(np.nanmax(maps[n].values) for n in maps)
    for ax, name in zip(axes, ("VL", "LM")):
        piv = maps[name]
        im = ax.imshow(piv.values, cmap="magma_r", vmin=0, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=7.5)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xlabel("Chroma (C) → насыщеннее")
        ax.set_ylabel("Value (V) → светлее")
        ax.set_title(f"{name} · среднее {np.nanmean(maps[name].values):.2f} ΔE", fontsize=11.5)
        plt.colorbar(im, ax=ax, label="ΔE (CAM16-LCD)")
    fig.suptitle("Ошибка линейного декодирования координат CAM16-LCD по телу Манселла\n"
                 "(тестовые цвета, 5-фолд по цветам; аналог Fig. 3 LIM26)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig7_error_heatmap_dE.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    df, VL, LM = mm.load()
    groups = mm.build_pairs(df)
    layers = {"VL": VL, "LM": LM}
    colors = {"VL": ACCENT_D, "LM": ACCENT}

    if "--heatmap-only" in sys.argv:
        print("== error heatmap (dE, coordinate decoding) ==")
        error_heatmap_dE(df, layers)
        print("saved fig7_error_heatmap_dE")
        return

    print("== test-size sweep: Munsell ==")
    mres = munsell_sweep(df, groups, layers)
    print("== test-size sweep: Leeds ==")
    lres = leeds_sweep()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=False)
    for name in ("VL", "LM"):
        mu = [np.nanmean(mres[name][f]) for f in FRACS]
        sdv = [np.nanstd(mres[name][f]) for f in FRACS]
        a1.errorbar(FRACS, mu, yerr=sdv, fmt="o-", color=colors[name],
                    lw=2, capsize=3, label=name)
        mu = [np.nanmean(lres[name][f]) for f in FRACS]
        sdv = [np.nanstd(lres[name][f]) for f in FRACS]
        a2.errorbar(FRACS, mu, yerr=sdv, fmt="o-", color=colors[name],
                    lw=2, capsize=3, label=name)
    a1.axhline(0.288, color=GRAY_D, ls="--", lw=1.2)
    a1.text(0.97, 0.293, "CAM16-LCD", ha="right", color=GRAY_D, fontsize=8.5,
            transform=a1.get_yaxis_transform())
    a1.set_title("Манселл (сплит по цветам, group-k)", fontsize=11.5)
    a2.axhline(0.256, color=GRAY_D, ls="--", lw=1.2)
    a2.text(0.97, 0.259, "CAM16-SCD", ha="right", color=GRAY_D, fontsize=8.5,
            transform=a2.get_yaxis_transform())
    a2.set_title("COMBVD-Leeds (сплит по центрам)", fontsize=11.5)
    a2.text(0.5, 0.02, "f=0.1 и f≥0.7 недостижимы: сплит по центрам\nотбрасывает «мостиковые» пары",
            transform=a2.transAxes, ha="center", fontsize=8, color="#888")
    for ax in (a1, a2):
        ax.set_xlim(0.05, 0.95)
        ax.set_xlabel("Доля тестовой выборки")
        ax.set_ylabel("STRESS на тесте")
        ax.legend(frameon=False); ax.grid(color="#EEE")
    fig.suptitle("Зависимость качества карты от размера тестовой выборки "
                 f"(m={M}, {len(SEEDS)} сида, mean±std)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig6_testsize.{ext}", bbox_inches="tight")
    plt.close(fig)

    print("== error heatmap ==")
    err_maps = error_heatmap(df, groups, layers)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    vmax = max(np.nanmax(err_maps[n].values) for n in err_maps)
    for ax, name in zip(axes, ("VL", "LM")):
        piv = err_maps[name]
        im = ax.imshow(piv.values, cmap="magma_r", vmin=0, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=7.5)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xlabel("Chroma (C) → насыщеннее")
        ax.set_ylabel("Value (V) → светлее")
        ax.set_title(f"Карта · {name}", fontsize=11.5)
        plt.colorbar(im, ax=ax, label="средняя |k·d̂ − 1|")
    fig.suptitle("Анализ ошибок карты по цветовому телу Манселла: отклонение "
                 "воспроизведения единичного шага (тестовые пары, 5-фолд)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig7_error_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)

    print("== error heatmap (dE, coordinate decoding) ==")
    error_heatmap_dE(df, layers)
    print("saved fig6_testsize, fig7_error_heatmap, fig7_error_heatmap_dE")


if __name__ == "__main__":
    main()
