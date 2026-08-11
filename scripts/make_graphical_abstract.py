"""
Графический абстракт (fig0), версия 4.

Слева: цветовые центры Манселла (1755, реальные sRGB-цвета) в ТРЁХМЕРНЫХ
координатах CAM16-UCS (a′, b′, J′) + рёбра соседних пар цепочек. Трёхмерное
представление разделяет цвета разной светлоты, которые в плоскости a′b′
сливались. Прямоугольная рамка выделяет область зелёных цветов; вынесенная
выноска показывает ДВЕ пары из этой области с общим якорем 7.5G 7/4 — одну с
различием по насыщенности (7/4 ↔ 7/6), другую по светлоте (7/4 ↔ 8/4). Для
каждой пары — три отрезка в одном масштабе: «по данным» (1 шаг Манселла),
предсказание CAM16-UCS и предсказание линейного отображения (оба после групповой
калибровки k своей оси).

Справа: STRESS-гистограммы «отображение vs аналитическая колориметрия» на
Манселле (group-k, подмножество 1755) и Leeds (сплит по центрам) + пол шума
человека.

Прозрачность: гистограммы читают data/analysis/{map_munsell_groupk,
verify_combvd_leak}.json; параметры зум-пар пишутся в data/analysis/fig0_abstract.json.
Отображение для зумов: VL, m=256, объектный 80/20 сплит (протокол
map_munsell_group_stress); групповой k каждой оси считается по тестовым парам.
"""
import importlib.util
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from colour import xyY_to_XYZ, XYZ_to_sRGB
from colour.models import XYZ_to_CAM16UCS

RNG = 42
OUT = Path("graphics")
A = Path("data/analysis")
rcParams.update({
    "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 140,
    "font.family": "DejaVu Sans",
})
ACCENT, ACCENT_D = "#3B6FB5", "#26456f"
GRAY, GRAY_D = "#9AA0A6", "#5F6368"
FLOOR, WARN, GT = "#2E9E6B", "#C77B30", "#333333"
PAIR = "#D62728"          # сравниваемые пары в приближении зелёной области
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

    # mapping: VL, m=256, object-wise 80/20 (protocol of analyze_representations)
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

    def d_cam(i, j):
        return float(np.linalg.norm(ucs[i] - ucs[j]))

    def d_map(i, j):
        return float(np.linalg.norm((Xs[i] - Xs[j]) @ Amap.T))

    def group_k(g):                                 # per-axis group-k on test pairs
        tep = sub(groups[g], te)
        dc = np.linalg.norm(ucs[tep[:, 0]] - ucs[tep[:, 1]], axis=1)
        dm = np.linalg.norm((Xs[tep[:, 0]] - Xs[tep[:, 1]]) @ Amap.T, axis=1)
        return k_opt(dc), k_opt(dm)

    def find(H, V, C):
        q = df[(df.H == H) & (df.V == V) & (df.C == C)]
        return int(q.index[0])

    anchor = find("7.5G", 7, 4)
    hue_j = find("5.0G", 7, 4)                      # шаг по тону (varying-H)
    bri_j = find("7.5G", 8, 4)                      # шаг по светлоте (varying-V)
    sat_j = find("7.5G", 7, 6)                      # шаг по насыщенности (varying-C)
    kh_cam, kh_map = group_k("varying-H")
    kv_cam, kv_map = group_k("varying-V")
    kc_cam, kc_map = group_k("varying-C")

    def pack(i, j, k_cam, k_map):
        return dict(i=int(i), j=int(j),
                    munsell_i=f"{df.loc[i,'H']} {df.loc[i,'V']}/{df.loc[i,'C']}",
                    munsell_j=f"{df.loc[j,'H']} {df.loc[j,'V']}/{df.loc[j,'C']}",
                    ratio_cam=float(k_cam * d_cam(i, j)),
                    ratio_map=float(k_map * d_map(i, j)))

    zoom = dict(
        anchor_munsell=f"{df.loc[anchor,'H']} {df.loc[anchor,'V']}/{df.loc[anchor,'C']}",
        box_idx=[int(anchor), int(hue_j), int(bri_j), int(sat_j)],
        hue=pack(anchor, hue_j, kh_cam, kh_map),     # H
        bri=pack(anchor, bri_j, kv_cam, kv_map),     # V
        sat=pack(anchor, sat_j, kc_cam, kc_map))     # C
    return df, groups, ucs, srgb, zoom


# ---------------------------------------------------------- 3-D wireframe box
def draw_box3d(ax, pts_abJ, pad, color):
    lo = pts_abJ.min(0) - pad
    hi = pts_abJ.max(0) + pad
    corners = np.array(list(itertools.product(*zip(lo, hi))))   # 8 × 3
    for a, b in itertools.combinations(range(8), 2):
        if np.sum(corners[a] != corners[b]) == 1:               # edge = 1 coord differs
            seg = np.array([corners[a], corners[b]])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=1.7, zorder=6)


# --------------------------------------------------------------------- figure
def main():
    df, groups, ucs, srgb, zoom = compute_zoom_data()
    mapg = json.load(open(A / "map_munsell_groupk.json"))

    cam = mapg["cam16_on_subset"]
    def gk(v):                                       # group-k mean of a variant
        pg = cam[v]
        return (pg["varying-V"] + pg["varying-C"] + pg["varying-H"]) / 3
    mun_bars = [
        ("CAM16-LCD", gk("CAM16-LCD"), GRAY_D),
        ("CAM16-UCS", gk("CAM16-UCS"), GRAY_D),
        ("CAM16-SCD", gk("CAM16-SCD"), GRAY_D),
        ("CIEDE2000", gk("CIEDE2000"), WARN),
        ("отображение·LM", mapg["map"]["LM"]["256"]["group_k_mean"], ACCENT),
        ("отображение·VL", mapg["map"]["VL"]["256"]["group_k_mean"], ACCENT_D),
    ]

    fig = plt.figure(figsize=(12, 5.0))
    ax3d = fig.add_axes([0.00, 0.05, 0.37, 0.90], projection="3d")
    axz = fig.add_axes([0.395, 0.05, 0.285, 0.90])        # green zoom (3 pairs)
    axb1 = fig.add_axes([0.745, 0.22, 0.235, 0.56])       # bars: Манселл (+CIEDE2000)

    # ---------------- 3-D scatter + chains ----------------
    segs = []
    for g in groups:
        pr = groups[g]
        p0 = ucs[pr[:, 0]][:, [1, 2, 0]]                  # (a', b', J')
        p1 = ucs[pr[:, 1]][:, [1, 2, 0]]
        segs.append(np.stack([p0, p1], axis=1))
    ax3d.add_collection3d(Line3DCollection(np.concatenate(segs), colors="#AAAAAA",
                                           linewidths=0.6, alpha=0.6))
    ax3d.scatter(ucs[:, 1], ucs[:, 2], ucs[:, 0], c=srgb, s=6, lw=0,
                 depthshade=False)
    ax3d.set_xlabel("a′", fontsize=14, labelpad=2)
    ax3d.set_ylabel("b′", fontsize=14, labelpad=2)
    ax3d.set_zlabel("J′ (светлота)", fontsize=14, labelpad=4)
    ax3d.tick_params(labelsize=10)
    ax3d.view_init(elev=16, azim=-60)
    ax3d.set_box_aspect((1, 1, 0.9))

    # rectangular frame around the highlighted green region
    box_pts = ucs[zoom["box_idx"]][:, [1, 2, 0]]          # (a', b', J')
    box_pad = np.array([4.0, 4.0, 5.0])
    draw_box3d(ax3d, box_pts, pad=box_pad, color=FLOOR)

    # ---- magnified copy of the framed region, in the free zone on top ----
    from mpl_toolkits.mplot3d import proj3d
    fig.canvas.draw()
    bc = box_pts.mean(0)
    xb, yb, _ = proj3d.proj_transform(bc[0], bc[1], bc[2], ax3d.get_proj())
    bx_fig = fig.transFigure.inverted().transform(ax3d.transData.transform((xb, yb)))

    axmag = fig.add_axes([0.075, 0.685, 0.147, 0.259], projection="3d")   # ~30% меньше
    P = ucs[:, [1, 2, 0]]
    # тесное поле приближения: сравниваемые цвета отстоят от якоря на один шаг
    # (разброс по b′ всего ≈1.9), при широком поле пары сливались в точку
    mag_pad = np.array([2.2, 1.8, 2.8])
    lo = box_pts.min(0) - mag_pad
    hi = box_pts.max(0) + mag_pad
    sel = np.all((P >= lo) & (P <= hi), axis=1)
    key = np.array(zoom["box_idx"])                      # [anchor, hue, bri, sat]
    anchor_i, neigh = int(key[0]), key[1:]
    keymask = np.zeros(len(df), bool); keymask[key] = True
    ctx = sel & ~keymask
    locsegs = []
    for g in groups:
        pr = groups[g]
        m = sel[pr[:, 0]] & sel[pr[:, 1]]
        if m.any():
            locsegs.append(np.stack([P[pr[m, 0]], P[pr[m, 1]]], axis=1))
    if locsegs:
        axmag.add_collection3d(Line3DCollection(np.concatenate(locsegs),
                                                colors="#CCCCCC", linewidths=0.6))
    axmag.scatter(P[ctx, 0], P[ctx, 1], P[ctx, 2], c=srgb[ctx], s=26, lw=0,
                  alpha=0.5, depthshade=False)
    draw_box3d(axmag, box_pts, pad=mag_pad, color=FLOOR)
    vpad = mag_pad * 1.20                       # рамка — чуть внутри поля зрения
    vlo, vhi = box_pts.min(0) - vpad, box_pts.max(0) + vpad
    axmag.set_xlim(vlo[0], vhi[0])
    axmag.set_ylim(vlo[1], vhi[1])
    axmag.set_zlim(vlo[2], vhi[2])
    # три сравниваемые пары «якорь ↔ сосед» — красные рёбра поверх серых цепочек
    pair_segs = np.stack([np.repeat(P[[anchor_i]], len(neigh), axis=0), P[neigh]],
                         axis=1)
    axmag.add_collection3d(Line3DCollection(pair_segs, colors=PAIR,
                                            linewidths=2.4, zorder=6))
    # соседи, затем якорь-звезда — последней, чтобы её не перекрывали маркеры
    axmag.scatter(P[neigh, 0], P[neigh, 1], P[neigh, 2], c=srgb[neigh], s=85,
                  edgecolors=PAIR, linewidths=1.8, depthshade=False, zorder=7)
    axmag.scatter([P[anchor_i, 0]], [P[anchor_i, 1]], [P[anchor_i, 2]],
                  c=[srgb[anchor_i]], marker="*", s=300, edgecolors=PAIR,
                  linewidths=1.6, depthshade=False, zorder=8)
    axmag.view_init(elev=16, azim=-60)
    axmag.set_box_aspect((1, 1, 0.9))
    axmag.set_xticks([]); axmag.set_yticks([]); axmag.set_zticks([])
    axmag.text2D(0.5, 1.02, "приближение выделенной области",
                 transform=axmag.transAxes, ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold", color="#1c6b49")
    # legend for the three compared points (+ anchor) beside the inset
    # чуть левее оси J′, чтобы подложка не накрывала верхнюю метку шкалы
    axleg = fig.add_axes([0.192, 0.655, 0.166, 0.275]); axleg.axis("off")
    axleg.set_xlim(0, 1); axleg.set_ylim(0, 1)
    # полупрозрачная подложка, чтобы легенда читалась поверх облака точек
    axleg.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                   boxstyle="round,pad=0.018", fc="white",
                                   ec="#C9D6D0", lw=0.8, alpha=0.85, zorder=0,
                                   transform=axleg.transAxes))
    items = [("*", srgb[anchor_i], 260, "7.5G 7/4  (якорь)"),
             ("o", srgb[neigh[0]], 95, "5.0G 7/4 — тон"),
             ("o", srgb[neigh[1]], 95, "7.5G 8/4 — светлота"),
             ("o", srgb[neigh[2]], 95, "7.5G 7/6 — насыщенность")]
    for r, (mk, col, sz, txt) in enumerate(items):
        y = 0.86 - r * 0.235
        axleg.scatter([0.09], [y], marker=mk, s=sz, c=[col], edgecolors=PAIR,
                      linewidths=1.4, zorder=3)
        axleg.text(0.22, y, txt, va="center", ha="left", fontsize=9.5, color="#222",
                   zorder=3)
    # magnifier leader lines from the framed region to the inset bottom corners
    for corner in [(0.075, 0.685), (0.222, 0.685)]:
        fig.add_artist(FancyArrowPatch(bx_fig, corner, transform=fig.transFigure,
                                       arrowstyle="-", color="#9DBFAE", lw=1.0,
                                       linestyle="--", zorder=0))

    # ---------------- green zoom inset (two pairs) ----------------
    axz.set_xlim(0, 1); axz.set_ylim(0, 1); axz.axis("off")
    axz.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                                 boxstyle="round,pad=0.012", fc="#F6FBF8",
                                 ec=FLOOR, lw=1.8, zorder=0, transform=axz.transAxes))
    axz.text(0.06, 0.965, f"Зелёная область ({zoom['anchor_munsell']})",
             fontsize=13, fontweight="bold", color="#1c6b49")

    x0, L = 0.46, 0.28                               # базовая длина = 1 шаг

    def block(y_sub, sub_title, pair):
        i, j = pair["i"], pair["j"]
        axz.text(0.06, y_sub, sub_title, fontsize=11.5, fontweight="bold", color="#333")
        rows = [("по данным", 1.0, GT, True),
                ("CAM16-UCS", pair["ratio_cam"], GRAY_D, False),
                ("отображение", pair["ratio_map"], ACCENT_D, False)]
        y0 = y_sub - 0.070
        for r, (lab, val, col, is_gt) in enumerate(rows):
            y = y0 - r * 0.056
            axz.text(x0 - 0.03, y, lab, ha="right", va="center", fontsize=10.5,
                     color="#333")
            axz.plot([x0, x0 + L * val], [y, y], color=col, lw=4.5,
                     solid_capstyle="butt", zorder=3)
            if is_gt:
                axz.scatter([x0, x0 + L * val], [y, y], s=95,
                            c=[srgb[i], srgb[j]], edgecolors="#333",
                            linewidths=0.8, zorder=4)
                lab_v = "1 шаг"
            else:
                lab_v = f"{val:.2f}"
            axz.text(x0 + L * max(val, 1.0) + 0.02, y, lab_v, va="center",
                     fontsize=11, color=col,
                     fontweight="bold" if col == ACCENT_D else "normal")
        axz.plot([x0 + L, x0 + L], [y0 - 0.135, y0 + 0.028], ls=":", lw=1.0,
                 color="#999", zorder=2)

    block(0.895, "по тону (H):  7.5G ↔ 5.0G", zoom["hue"])
    axz.axhline(0.655, xmin=0.06, xmax=0.94, color="#DDE7E1", lw=1.2)
    block(0.605, "по светлоте (V):  7/4 ↔ 8/4", zoom["bri"])
    axz.axhline(0.365, xmin=0.06, xmax=0.94, color="#DDE7E1", lw=1.2)
    block(0.315, "по насыщенности (C):  7/4 ↔ 7/6", zoom["sat"])

    # (зелёная стрелка-выноска к зум-блоку убрана: перекрывала подпись оси J′)

    # ---------------- bars ----------------
    def bars(ax, data, label, ymax, noise_line=False):
        x = np.arange(len(data))
        vals = [d[1] for d in data]
        ax.bar(x, vals, color=[d[2] for d in data], width=0.62, zorder=3,
               edgecolor="white", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi, v + ymax * 0.025, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=11, color="#222")
        if noise_line:
            ax.axhline(NOISE, ls="--", lw=1.3, color=FLOOR, zorder=2)
            ax.plot([3.25, 3.6], [0.40, 0.40], ls="--", lw=1.3, color=FLOOR,
                    clip_on=False)
            ax.text(3.7, 0.40, "пол шума человека 0.090", va="center",
                    ha="left", color=FLOOR, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in data], fontsize=9, rotation=28, ha="right")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("STRESS ↓", fontsize=13)
        ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", color="#333")
        ax.grid(axis="y", color="#EEE", zorder=0)
        ax.tick_params(labelsize=11)

    bars(axb1, mun_bars, "Надпороговые", 0.50)
    yb = 0.445
    axb1.plot([0, 0, 5, 5], [0.35, yb, yb, 0.15], color=ACCENT_D, lw=1.1)
    axb1.text(2.5, yb + 0.008, "в 2.8 раза ближе к человеку", color=ACCENT_D,
              fontsize=12, ha="center", fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig0_graphical_abstract.{ext}", bbox_inches="tight")
    plt.close(fig)

    json.dump({
        "script": "scripts/make_graphical_abstract.py",
        "date": str(date.today()),
        "protocol": "3D CAM16-UCS (a',b',J'); две зелёные зум-пары с якорем 7.5G 7/4 "
                    "(по насыщенности 7/4↔7/6 = varying-C, по светлоте 7/4↔8/4 = "
                    "varying-V); отображение VL m=256, объектный 80/20; групповой k "
                    "каждой оси по тестовым парам, тот же k и для CAM16-UCS, и для "
                    "отображения; гистограммы — из map_munsell_groupk.json и "
                    "verify_combvd_leak.json",
        "zoom_pairs_green": {"anchor": zoom["anchor_munsell"], "hue": zoom["hue"],
                             "bri": zoom["bri"], "sat": zoom["sat"]},
        "bars": {"munsell": {n: round(v, 4) for n, v, _ in mun_bars}},
    }, open(A / "fig0_abstract.json", "w"), indent=2, ensure_ascii=False)
    print("saved fig0_graphical_abstract + data/analysis/fig0_abstract.json")
    for tag in ("hue", "bri", "sat"):
        z = zoom[tag]
        print(f"  {tag}: {z['munsell_i']} <-> {z['munsell_j']}  "
              f"CAM16 {z['ratio_cam']:.2f}  map {z['ratio_map']:.2f}")


if __name__ == "__main__":
    main()
