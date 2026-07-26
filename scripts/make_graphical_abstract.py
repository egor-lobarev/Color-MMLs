"""
Графический абстракт (fig0), версия 2.

Слева: цветовые центры Манселла (1755, реальные sRGB-цвета) в плоскости a'b'
CAM16-UCS + рёбра соседних пар цепочек. Два «зума» вынесены в сторону: конкретная
пара центров и три отрезка в одном масштабе — «по данным» (1 шаг Манселла),
предсказание CAM16-UCS и предсказание нашей карты (оба после честной групповой
калибровки k). Тоновая пара — CAM16 промахивается, карта держит единицу;
светлотная — оба точны.

Справа: STRESS-гистограммы «мы vs аналитическая колориметрия» на Манселле
(group-k, подмножество 1755) и Leeds (сплит по центрам) + пол шума человека.

Прозрачность: гистограммы читают data/analysis/{map_munsell_groupk,
verify_combvd_leak}.json; параметры зум-пар пишутся в data/analysis/fig0_abstract.json.
Карта для зумов: VL, m=256, объектный 80/20 сплит (протокол map_munsell_group_stress),
зум-пары выбираются только из тестовых.
"""
import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from colour import xyY_to_XYZ, XYZ_to_sRGB
from colour.models import XYZ_to_CAM16UCS

RNG = 42
OUT = Path("graphics")
A = Path("data/analysis")
rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 140,
    "font.family": "DejaVu Sans",
})
ACCENT, ACCENT_D = "#3B6FB5", "#26456f"
GRAY, GRAY_D = "#9AA0A6", "#5F6368"
FLOOR, WARN, GT = "#2E9E6B", "#C77B30", "#333333"
NOISE = 0.090


def load_module(p):
    s = importlib.util.spec_from_file_location(Path(p).stem, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


gs_mod = load_module("scripts/analyze_gram_spectrum.py")
mm = gs_mod.mm


# --------------------------------------------------------------- zoom-pair data
def compute_zoom_data():
    np.random.seed(RNG); torch.manual_seed(RNG)
    df, VL, LM = mm.load()
    groups = mm.build_pairs(df)
    XYZ = xyY_to_XYZ(df[["x", "y", "Y"]].to_numpy(float) * np.array([1, 1, 1 / 100]))
    ucs = XYZ_to_CAM16UCS(XYZ)                     # (J', a', b')
    srgb = np.clip(XYZ_to_sRGB(XYZ), 0, 1)

    # map: VL, m=256, object-wise 80/20 (protocol of analyze_representations)
    X = VL
    sd = X.std(0); sd[sd == 0] = 1; Xs = X / sd
    idx = np.random.RandomState(RNG).permutation(len(df))
    tr = set(idx[: int(0.8 * len(df))].tolist())
    te = set(idx[int(0.8 * len(df)):].tolist())

    def sub(pr, S):
        return pr[[a in S and b in S for a, b in pr]]

    train_pairs = np.vstack([sub(groups[g], tr) for g in groups])
    Amap, _ = gs_mod.train_A(Xs, train_pairs, 256, seed=RNG)

    def k_opt(d):                                   # argmin ||k d - 1||
        return d.sum() / (d ** 2).sum()

    zoom = {}
    for g in ("varying-H", "varying-V"):
        tep = sub(groups[g], te)
        d_cam = np.linalg.norm(ucs[tep[:, 0]] - ucs[tep[:, 1]], axis=1)
        dx = Xs[tep[:, 0]] - Xs[tep[:, 1]]
        d_map = np.linalg.norm(dx @ Amap.T, axis=1)
        k_cam, k_map = k_opt(d_cam), k_opt(d_map)
        r_cam, r_map = k_cam * d_cam, k_map * d_map
        if g == "varying-H":
            # типичный промах CAM16 по тону: |r-1| около медианы промахов,
            # карта близка к 1, пара хроматическая (видна на плоскости a'b')
            med = np.median(np.abs(r_cam - 1))
            ok = (np.abs(r_map - 1) < 0.06) & (d_cam > 4)
            cand = np.where(ok)[0]
            pick = cand[np.argmin(np.abs(np.abs(r_cam[cand] - 1) - med))]
        else:
            # светлотная пара: оба точны; берём хроматическую, чтобы цвета видны
            C = df["C"].to_numpy()
            ok = (np.abs(r_map - 1) < 0.04) & (np.abs(r_cam - 1) < 0.04) \
                 & (C[tep[:, 0]] >= 6)
            cand = np.where(ok)[0]
            pick = cand[len(cand) // 2]
        i, j = tep[pick]
        zoom[g] = dict(
            i=int(i), j=int(j),
            munsell_i=f"{df.loc[i,'H']} {df.loc[i,'V']}/{df.loc[i,'C']}",
            munsell_j=f"{df.loc[j,'H']} {df.loc[j,'V']}/{df.loc[j,'C']}",
            ratio_cam=float(r_cam[pick]), ratio_map=float(r_map[pick]),
            k_cam=float(k_cam), k_map=float(k_map),
            median_abs_err_cam=float(np.median(np.abs(r_cam - 1))),
            median_abs_err_map=float(np.median(np.abs(r_map - 1))))
    return df, groups, ucs, srgb, zoom


# --------------------------------------------------------------------- figure
def main():
    df, groups, ucs, srgb, zoom = compute_zoom_data()
    mapg = json.load(open(A / "map_munsell_groupk.json"))
    leak = json.load(open(A / "verify_combvd_leak.json"))

    cam = mapg["cam16_on_subset"]
    def gk(v):                                       # group-k mean of a variant
        pg = cam[v]
        return (pg["varying-V"] + pg["varying-C"] + pg["varying-H"]) / 3
    mun_bars = [
        ("CAM16-LCD", gk("CAM16-LCD"), GRAY_D),
        ("CAM16-UCS", gk("CAM16-UCS"), GRAY_D),
        ("CAM16-SCD", gk("CAM16-SCD"), GRAY_D),
        ("МЯМ · LM", mapg["map"]["LM"]["256"]["group_k_mean"], ACCENT),
        ("МЯМ · VL", mapg["map"]["VL"]["256"]["group_k_mean"], ACCENT_D),
    ]
    bl = leak["baselines"]
    leeds_bars = [
        ("CAM16-LCD", bl["CAM16-LCD (dE)"], GRAY_D),
        ("CAM16-UCS", bl["CAM16-UCS (dE)"], GRAY_D),
        ("CAM16-SCD", bl["CAM16-SCD (dE)"], GRAY_D),
        ("CIEDE2000", bl["CIEDE2000"], WARN),
        ("МЯМ · LM", leak["map"]["LM"]["by_center"]["1024"]["mean"], ACCENT),
        ("МЯМ · VL", leak["map"]["VL"]["by_center"]["256"]["mean"], ACCENT_D),
    ]

    fig = plt.figure(figsize=(14.6, 5.6))
    # осевые области: [x0, y0, w, h]
    axs = fig.add_axes([0.045, 0.10, 0.30, 0.74])        # scatter
    axz1 = fig.add_axes([0.375, 0.50, 0.215, 0.34])      # zoom: тон
    axz2 = fig.add_axes([0.375, 0.08, 0.215, 0.34])      # zoom: светлота
    axb1 = fig.add_axes([0.665, 0.575, 0.325, 0.275])    # bars: Манселл
    axb2 = fig.add_axes([0.665, 0.10, 0.325, 0.275])     # bars: Leeds

    # ---------------- scatter + chains ----------------
    segs = []
    for g in groups:
        pr = groups[g]
        segs.append(np.stack([ucs[pr[:, 0]][:, 1:], ucs[pr[:, 1]][:, 1:]], axis=1))
    axs.add_collection(LineCollection(np.concatenate(segs), colors="#CCCCCC",
                                      linewidths=0.35, alpha=0.55, zorder=1))
    axs.scatter(ucs[:, 1], ucs[:, 2], s=7, c=srgb, lw=0, zorder=2)
    axs.set_aspect("equal")
    axs.set_xlabel("CAM16-UCS a′", fontsize=9.5)
    axs.set_ylabel("CAM16-UCS b′", fontsize=9.5)
    axs.set_title("Цветовые центры Манселла и единичные\nперцептивные шаги (цепочки H/C/V)",
                  fontsize=10.5, pad=6)
    axs.tick_params(labelsize=8)

    # ---------------- zoom insets ----------------
    def zoom_inset(ax, g, title, note):
        z = zoom[g]
        i, j = z["i"], z["j"]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                                    boxstyle="round,pad=0.012", fc="#FAFAFA",
                                    ec="#BBB", lw=1.0, zorder=0,
                                    transform=ax.transAxes))
        ax.text(0.05, 0.90, title, fontsize=9.5, fontweight="bold", color="#222")
        ax.text(0.97, 0.90, f"{z['munsell_i']} ↔ {z['munsell_j']}",
                fontsize=7.8, color="#666", ha="right")
        x0, L = 0.30, 0.42                       # базовая длина = 1 шаг
        rows = [("по данным", 1.0, GT, True),
                ("CAM16-UCS", z["ratio_cam"], GRAY_D, False),
                ("карта МЯМ", z["ratio_map"], ACCENT_D, False)]
        for r, (lab, val, col, is_gt) in enumerate(rows):
            y = 0.68 - r * 0.235
            ax.text(x0 - 0.03, y, lab, ha="right", va="center", fontsize=8.8,
                    color="#333")
            ax.plot([x0, x0 + L * val], [y, y], color=col, lw=4.5,
                    solid_capstyle="butt", zorder=3)
            if is_gt:                            # концы = реальные цвета пары
                ax.scatter([x0, x0 + L * val], [y, y], s=110,
                           c=[srgb[i], srgb[j]], edgecolors="#333",
                           linewidths=0.8, zorder=4)
                lab_v = "1 шаг"
            else:
                lab_v = f"{val:.2f}"
            ax.text(x0 + L * max(val, 1.0) + 0.035, y, lab_v, va="center",
                    fontsize=8.8, color=col,
                    fontweight="bold" if col == ACCENT_D else "normal")
            if is_gt:                            # пунктир-ориентир единицы
                ax.plot([x0 + L, x0 + L], [0.13, 0.78], ls=":", lw=0.9,
                        color="#999", zorder=2)
        ax.text(0.05, 0.06, note, fontsize=8.2, color="#555")
        return i, j

    i1, j1 = zoom_inset(axz1, "varying-H", "Тоновая пара",
                        "CAM16 растягивает шаг по тону — карта держит единицу")
    i2, j2 = zoom_inset(axz2, "varying-V", "Светлотная пара",
                        "по светлоте CAM16 уже точен — карта его повторяет")

    for (i, j), axz, yz in [((i1, j1), axz1, 0.5), ((i2, j2), axz2, 0.5)]:
        mid = ucs[[i, j], 1:].mean(0)
        axs.scatter(ucs[[i, j], 1], ucs[[i, j], 2], s=46, facecolors="none",
                    edgecolors="#333", linewidths=1.1, zorder=3)
        fig.add_artist(ConnectionPatch(
            xyA=mid, coordsA=axs.transData, xyB=(0.015, yz),
            coordsB=axz.transAxes, color="#999", lw=0.9, ls="-", zorder=1))

    # ---------------- bars ----------------
    def bars(ax, data, title, ymax, noise_line=False):
        x = np.arange(len(data))
        vals = [d[1] for d in data]
        ax.bar(x, vals, color=[d[2] for d in data], width=0.62, zorder=3,
               edgecolor="white", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi, v + ymax * 0.025, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.2, color="#222")
        if noise_line:
            ax.axhline(NOISE, ls="--", lw=1.2, color=FLOOR, zorder=2)
            # подпись — в свободной зоне над столбцами, с образцом пунктира
            ax.plot([2.62, 3.02], [0.378, 0.378], ls="--", lw=1.2, color=FLOOR,
                    clip_on=False)
            ax.text(3.12, 0.378, "пол шума человека 0.090", va="center",
                    ha="left", color=FLOOR, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in data], fontsize=8.2)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("STRESS ↓", fontsize=9)
        ax.set_title(title, fontsize=10.5, pad=5, loc="left")
        ax.grid(axis="y", color="#EEE", zorder=0)
        ax.tick_params(labelsize=8)

    bars(axb1, mun_bars, "Надпороговые · Манселл (group-k, те же цвета)", 0.42)
    bars(axb2, leeds_bars, "Пороговые · Leeds (сплит по центрам)", 0.42,
         noise_line=True)
    # скобка «×2.8» между лучшим CAM16 и картой VL, над столбцами
    yb = 0.365
    axb1.plot([0, 0, 4, 4], [0.325, yb, yb, 0.14], color=ACCENT_D, lw=1.0)
    axb1.text(2.0, yb + 0.008, "в 2.8 раза ближе к человеку", color=ACCENT_D,
              fontsize=8.6, ha="center", fontweight="bold")

    fig.suptitle(
        "Внутренние представления МЯМ воспроизводят метрику цветовых различий человека:\n"
        "линейная карта из эмбеддингов превосходит каждый вариант CAM16 на его масштабе — "
        "модель не обучалась цветоразличению", fontsize=12, y=0.985)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig0_graphical_abstract.{ext}", bbox_inches="tight")
    plt.close(fig)

    json.dump({
        "script": "scripts/make_graphical_abstract.py",
        "date": str(date.today()),
        "protocol": "зумы: карта VL m=256, объектный 80/20, пары из теста; "
                    "групповой k и для CAM16-UCS, и для карты; гистограммы — из "
                    "map_munsell_groupk.json и verify_combvd_leak.json",
        "zoom_pairs": zoom,
        "bars": {"munsell": {n: round(v, 4) for n, v, _ in mun_bars},
                 "leeds": {n: round(v, 4) for n, v, _ in leeds_bars}},
    }, open(A / "fig0_abstract.json", "w"), indent=2, ensure_ascii=False)
    print("saved fig0_graphical_abstract + data/analysis/fig0_abstract.json")
    for g, z in zoom.items():
        print(f"  {g}: {z['munsell_i']} <-> {z['munsell_j']}  "
              f"CAM16 {z['ratio_cam']:.2f}  map {z['ratio_map']:.2f}  "
              f"(медианный |err| CAM16 {z['median_abs_err_cam']:.2f}, "
              f"map {z['median_abs_err_map']:.2f})")


if __name__ == "__main__":
    main()
