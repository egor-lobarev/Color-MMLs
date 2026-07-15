"""
Manifold analysis of MLLM color representations + analysis of the learned linear
map itself (Qwen2.5-VL-7B, Munsell 1755 colors).

Produces (graphics/):
  fig4_manifold.(png|pdf)     : (a) PCA cumulative explained variance (VL/LM),
                                (b) intrinsic dimension: Levina-Bickel MLE vs k
                                    + TwoNN estimates.
  fig5_map_spectrum.(png|pdf) : (a) singular spectrum of trained A (m=256),
                                (b) test STRESS vs SVD-truncation rank r  <- key:
                                    how many directions of the embedding carry
                                    the perceptual metric,
                                (c) input-dimension importance (column norms of A)
                                    + weight histogram (inset).

Reuses loaders/training from map_munsell_group_stress.py (same protocol:
object-wise split, group-k STRESS mean over varying-H/C/V).
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

RNG = 42
np.random.seed(RNG); torch.manual_seed(RNG)
OUT = Path("graphics"); OUT.mkdir(exist_ok=True)
rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 140,
    "font.family": "DejaVu Sans",
})
ACCENT, ACCENT_D, GRAY_D, FLOOR = "#3B6FB5", "#26456f", "#5F6368", "#2E9E6B"


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mm = load_module("scripts/map_munsell_group_stress.py")


# --------------------------------------------------- intrinsic dimension estimators
def id_mle_levina_bickel(X, k=20):
    """Levina-Bickel MLE on L2-normalized data (protocol of the notebook)."""
    Xn = normalize(X, norm="l2")
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(Xn)
    d, _ = nn.kneighbors(Xn)
    d = d[:, 1:]
    Tk = d[:, -1:]; Tj = d[:, :-1]
    eps = 1e-10
    denom = np.sum(np.log(np.maximum(Tk, eps) / np.maximum(Tj, eps)), axis=1)
    local = np.divide(k - 2, denom, out=np.full(len(denom), np.nan), where=denom > 0)
    return float(np.nanmean(local))  # duplicates (achromatic repeats) -> nan, drop


def id_twonn(X):
    """TwoNN (Facco et al., 2017): d = N / sum(log(r2/r1))."""
    nn = NearestNeighbors(n_neighbors=3, n_jobs=-1).fit(X)
    d, _ = nn.kneighbors(X)
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12)
    mu = mu[mu > 1.0]
    return float(len(mu) / np.sum(np.log(mu)))


# ------------------------------------------------------------------------- pipeline
def main():
    df, VL, LM = mm.load()
    groups = mm.build_pairs(df)
    layers = {"VL": VL, "LM": LM}
    colors = {"VL": ACCENT_D, "LM": ACCENT}
    print(f"colors={len(df)}, d={VL.shape[1]}")

    # ================= PCA + intrinsic dimension =================
    pca_res, id_res = {}, {}
    ks = [5, 10, 15, 20, 30, 50]
    for name, X in layers.items():
        p = PCA().fit(X - X.mean(0))
        evr = p.explained_variance_ratio_
        cum = np.cumsum(evr)
        lam = p.explained_variance_
        pr = float((lam.sum() ** 2) / (lam ** 2).sum())  # participation ratio
        n90, n95, n99 = [int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99)]
        pca_res[name] = dict(cum=cum, pr=pr, n90=n90, n95=n95, n99=n99)
        id_res[name] = dict(mle=[id_mle_levina_bickel(X, k) for k in ks],
                            twonn=id_twonn(X))
        print(f"[{name}] PR={pr:.1f}  n90={n90} n95={n95} n99={n99}  "
              f"TwoNN={id_res[name]['twonn']:.2f}  "
              f"MLE(k=20)={id_res[name]['mle'][ks.index(20)]:.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for name in layers:
        r = pca_res[name]
        a1.plot(np.arange(1, len(r["cum"]) + 1), r["cum"], color=colors[name],
                lw=2, label=f"{name}: 95% при {r['n95']} комп., PR={r['pr']:.0f}")
        a1.axvline(r["n95"], color=colors[name], ls=":", lw=1)
    a1.set_xscale("log")
    a1.axhline(0.95, color="#999", ls="--", lw=1)
    a1.set_xlabel("Число главных компонент (log)")
    a1.set_ylabel("Накопленная объяснённая дисперсия")
    a1.set_title("а) PCA-спектр эмбеддингов", fontsize=11.5)
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    a1.grid(color="#EEE")

    for name in layers:
        a2.plot(ks, id_res[name]["mle"], "o-", color=colors[name], lw=2,
                label=f"{name}: MLE; TwoNN = {id_res[name]['twonn']:.1f}")
    a2.axhline(3, color=FLOOR, ls="--", lw=1.2)
    a2.text(ks[0], 2.955, "3 = размерность цвета", ha="left", color=FLOOR, fontsize=9)
    a2.set_xlabel("k (число соседей)")
    a2.set_ylabel("Оценка внутренней размерности")
    a2.set_title("б) Внутренняя размерность (Levina–Bickel MLE)", fontsize=11.5)
    a2.legend(frameon=False, fontsize=9)
    a2.grid(color="#EEE")
    fig.suptitle("Многообразие цветовых представлений МЯМ: размерность ≈ 3 при "
                 "тысячах номинальных измерений", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_manifold.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ================= the linear map itself =================
    # object-wise 80/20 split; train A (m=256) on train pairs; SVD-truncate.
    n = len(df)
    idx = np.random.RandomState(RNG).permutation(n)
    tr, te = set(idx[: int(0.8 * n)].tolist()), set(idx[int(0.8 * n):].tolist())

    def sub(pr, S):
        return pr[[a in S and b in S for a, b in pr]]

    ranks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256]
    spec, curves, A_store = {}, {}, {}
    for name, X in layers.items():
        sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
        train_pairs = np.vstack([sub(groups[g], tr) for g in groups])
        A = mm.train_A(Xs, train_pairs, m=256).numpy()
        A_store[name] = (A, Xs)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        spec[name] = S
        st_r = []
        for r in ranks:
            Ar = torch.tensor((U[:, :r] * S[:r]) @ Vt[:r], dtype=torch.float32)
            pg = []
            for g in groups:
                tep = sub(groups[g], te)
                pg.append(mm.stress(mm.map_dist(Ar, Xs, tep), np.ones(len(tep))))
            st_r.append(np.mean(pg))
        curves[name] = st_r
        n95_energy = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.95) + 1)
        print(f"[{name}] map A: 95% энергии в {n95_energy} сингулярных числах; "
              f"STRESS(r=3)={st_r[ranks.index(3)]:.3f}, r=8: {st_r[ranks.index(8)]:.3f}, "
              f"full(256): {st_r[-1]:.3f}")

    fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for name in layers:
        S = spec[name]
        b1.plot(np.arange(1, len(S) + 1), S / S[0], color=colors[name], lw=2, label=name)
    b1.set_xscale("log"); b1.set_yscale("log")
    b1.set_xlabel("Номер сингулярного числа (log)")
    b1.set_ylabel("σᵢ / σ₁ (log)")
    b1.set_title("а) Спектр обученной карты A", fontsize=11.5)
    b1.legend(frameon=False); b1.grid(color="#EEE")

    for name in layers:
        b2.plot(ranks, curves[name], "o-", color=colors[name], lw=2, label=name)
    b2.axhline(0.288, color=GRAY_D, ls="--", lw=1.2)
    b2.text(ranks[-1], 0.295, "CAM16-LCD (0.288)", ha="right", color=GRAY_D, fontsize=8.5)
    b2.set_xscale("log")
    b2.set_xlabel("Ранг усечённой карты r (log)")
    b2.set_ylabel("STRESS на тесте (group-k)")
    b2.set_title("б) Качество vs ранг карты (SVD-усечение)", fontsize=11.5)
    b2.legend(frameon=False); b2.grid(color="#EEE")

    for name in layers:
        A, _ = A_store[name]
        cn = np.sort(np.linalg.norm(A, axis=0))[::-1]
        b3.plot(np.arange(1, len(cn) + 1), cn / cn[0], color=colors[name], lw=1.6, label=name)
    b3.set_xscale("log")
    b3.set_xlabel("Входные измерения эмбеддинга, ранжированные (log)")
    b3.set_ylabel("Норма столбца A (отн.)")
    b3.set_title("в) Значимость входных измерений", fontsize=11.5)
    b3.legend(frameon=False, loc="lower left"); b3.grid(color="#EEE")
    axins = b3.inset_axes([0.45, 0.45, 0.5, 0.45])
    axins.hist(A_store["VL"][0].ravel(), bins=120, color=ACCENT_D, alpha=0.85)
    axins.set_title("гистограмма весов A (VL)", fontsize=7.5)
    axins.tick_params(labelsize=6.5)

    fig.suptitle("Ядро перцептивной метрики компактно (ранга ≈8 достаточно, чтобы догнать CAM16-LCD),\n"
                 "но тонкая калибровка распределена по сотням направлений эмбеддинга", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_map_spectrum.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved fig4_manifold, fig5_map_spectrum")


if __name__ == "__main__":
    main()
