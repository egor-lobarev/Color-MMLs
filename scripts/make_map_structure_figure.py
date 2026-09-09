#!/usr/bin/env python3
"""Рис. 5 «Структура обученной метрики» — три панели вместо прежних четырёх фигур.

Заменяет fig4_manifold + fig5_map_spectrum + fig8_gram_spectrum +
fig9_gram_spectrum_joint (13 панелей) одной фигурой из трёх.

Причина укрупнения: панели дублировали друг друга. Собственные значения матрицы
Грама и сингулярные числа отображения связаны тождеством lambda_i(A^T A) =
sigma_i(A)^2, то есть «спектр Грама» и «сингулярный спектр A» — одна и та же
величина в двух видах. Функциональный ранг r_q — производная от кривой
SVD-усечения. PCA-спектр и нормы столбцов сводятся к нескольким числам в тексте.

Панели:
  (а) STRESS vs ранг SVD-усечения + накопленная дисперсия PCA на второй оси.
      Главная мысль блока: перцептивная метрика НЕ сводится к главным
      компонентам дисперсии — 95% дисперсии лежит в 6 компонентах (VL),
      а ранга 6 не хватает даже до уровня CAM16-LCD.
  (б) STRESS vs выходная размерность m, раздельное и совместное обучение.
      Мысль: при раздельном обучении визуальный энкодер заметно точнее
      языкового декодера, при совместном они сходятся. Слово «пол»
      сознательно не используется: в статье оно закреплено за полом шума
      человеческих измерений (0.090), это другая величина.
  (в) Спектр Грама (он же квадрат сингулярного спектра A) против полки шума
      инициализации. Мысль: у визуального энкодера сильных цветонесущих
      направлений в разы больше.

Источники (руками числа не правятся, см. §6.5 плана):
  data/analysis/gram_spectrum.{json,npz}         — раздельное обучение
  data/analysis/gram_spectrum_joint.{json,npz}   — совместное
  data/analysis/representations_fig4_fig5.json   — сводка PCA
  эмбеддинги Манселла                            — кривая PCA (пересчёт)

    python scripts/make_map_structure_figure.py
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
OUT = ROOT / "graphics"

rcParams.update({
    "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 140,
    "font.family": "DejaVu Sans",
})
# палитра серии: синие — МЯМ, серые — аналитика
VL_C, LM_C, GRAY_D, GRAY = "#26456f", "#3B6FB5", "#5F6368", "#9AA0A6"
CAM16_LCD = 0.288          # group-k на том же подмножестве цветов
# спектр строится при полной размерности: контраст слоёв живёт там
M_SPEC = 3584


def load_pca_curves() -> dict:
    """Накопленная объяснённая дисперсия по слоям (пересчёт из эмбеддингов)."""
    spec = importlib.util.spec_from_file_location(
        "mm", ROOT / "scripts" / "map_munsell_group_stress.py")
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    df, VL, LM = mm.load()
    out = {}
    for name, X in (("VL", VL), ("LM", LM)):
        X = np.asarray(X, dtype=np.float64)
        cum = np.cumsum(PCA().fit(X - X.mean(0)).explained_variance_ratio_)
        out[name] = cum
    return out


def main() -> int:
    gs = json.loads((A / "gram_spectrum.json").read_text())
    gj = json.loads((A / "gram_spectrum_joint.json").read_text())
    zs = np.load(A / "gram_spectrum.npz")
    zj = np.load(A / "gram_spectrum_joint.npz")
    reps = json.loads((A / "representations_fig4_fig5.json").read_text())

    print("пересчёт PCA-кривой из эмбеддингов…")
    pca = load_pca_curves()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # ---------------- (а) ранг усечения + PCA ----------------
    ax = axes[0]
    for lvl, col in (("VL", VL_C), ("LM", LM_C)):
        for src, z, ls, lab in ((gs, zs, "-", "раздельно"), (gj, zj, "--", "совместно")):
            c = z[f"curve_{lvl}_256"]
            ax.plot(np.arange(1, len(c) + 1), c, ls, color=col, lw=1.9,
                    label=f"{lvl}, {lab}")
    ax.axhline(CAM16_LCD, ls=":", lw=1.4, color=GRAY_D)
    ax.text(280, CAM16_LCD + 0.016, f"CAM16-LCD ({CAM16_LCD:.3f})",
            color=GRAY_D, fontsize=10.5, ha="right")

    # Сопоставление с PCA — без второй оси: она загромождала панель.
    # Мысль: почти вся дисперсия представления лежит в первых 6 компонентах,
    # и этого хватает, чтобы сравняться с вековой колориметрией; но тонкая
    # метрика набирается направлениями, на которые приходятся оставшиеся 5%.
    n95_vl = reps["fig4_manifold"]["VL"]["n95"]
    ax.axvspan(1, n95_vl, color=GRAY, alpha=0.12, lw=0)
    ax.text(1.6, 0.055, f"95% дисперсии\nпредставления\n({n95_vl} компонент)",
            color=GRAY_D, fontsize=9.5, va="bottom")
    c_vl = zs["curve_VL_256"]
    r_hit = int(np.argmax(c_vl <= CAM16_LCD) + 1)
    ax.annotate(f"ранг {r_hit}", xy=(r_hit, c_vl[r_hit - 1]), xytext=(16, 0.46),
                fontsize=10, color=VL_C,
                arrowprops=dict(arrowstyle="->", color=VL_C, lw=1.1))
    ax.set_xscale("log")
    ax.set_xlim(1, 300)
    ax.set_xlabel("Ранг усечённого отображения r")
    ax.set_ylabel("STRESS на тесте")
    ax.set_title("а) сколько направлений несут метрику", fontsize=12.5, loc="left")
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)

    # ---------------- (б) STRESS vs m ----------------
    ax = axes[1]
    for lvl, col in (("VL", VL_C), ("LM", LM_C)):
        for src, ls, lab in ((gs, "-", "раздельно"), (gj, "--", "совместно")):
            ms = sorted(int(k) for k in src[lvl] if k.isdigit())
            mu = [src[lvl][str(m)]["stress_mean"] for m in ms]
            sd = [src[lvl][str(m)]["stress_std"] for m in ms]
            ax.errorbar(ms, mu, yerr=sd, fmt=ls, marker="o", ms=3.5, lw=1.8,
                        color=col, capsize=2.5, label=f"{lvl}, {lab}")
    ax.axhline(CAM16_LCD, ls=":", lw=1.4, color=GRAY_D)
    ax.set_xscale("log")
    ax.set_xlabel("Выходная размерность m")
    ax.set_ylabel("STRESS на тесте")
    ax.set_title("б) предел качества по слоям", fontsize=12.5, loc="left")
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.grid(alpha=0.25)

    # ---------------- (в) спектр Грама ----------------
    # Строится при m = M_SPEC (полная размерность): контраст между слоями
    # существует именно там. При m = 256 над полкой шума инициализации нет
    # ни одного направления у обоих слоёв, хотя качество там лучшее —
    # проверено, см. numbers["above_shelf"].
    ax = axes[2]
    # Только раздельное обучение: при совместном регуляризация отключена,
    # масштаб A растёт свободно, и его спектр несопоставим с той же полкой
    # (полка уходит на ~4 порядка). Компактность совместной метрики видна
    # на панелях (а) и (б), дублировать её здесь нечем.
    shelf = {lvl: zs[f"lam0_{lvl}_{M_SPEC}"][0] for lvl in ("VL", "LM")}
    for lvl, col in (("VL", VL_C), ("LM", LM_C)):
        lam = zs[f"lam_{lvl}_{M_SPEC}"]
        ax.plot(np.arange(1, len(lam) + 1), lam / shelf[lvl], "-", color=col,
                lw=1.9, label=f"{lvl}")
    ax.axhline(1.0, ls=":", lw=1.5, color=GRAY_D)
    ax.text(120, 1.10, "полка шума инициализации", color=GRAY_D, fontsize=9.5)
    for lvl, col, dy in (("VL", VL_C, 5.0), ("LM", LM_C, 2.4)):
        n_above = int((zs[f"lam_{lvl}_{M_SPEC}"] > shelf[lvl]).sum())
        ax.annotate(f"{n_above} λ", xy=(n_above, 1.0), xytext=(n_above * 1.5, dy),
                    color=col, fontsize=10,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.15, 12)
    ax.set_xlabel("Номер собственного значения")
    ax.set_ylabel(r"$\lambda_i$ / полка шума")
    ax.set_title(f"в) веса направлений метрики (m = {M_SPEC})",
                 fontsize=12.5, loc="left")
    ax.legend(fontsize=10, framealpha=0.9, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_map_structure.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ---- числа, ушедшие из панелей в текст ----
    numbers = {
        "script": "scripts/make_map_structure_figure.py",
        "date": str(date.today()),
        "protocol": ("Манселл, group-k, 5-фолд CV по цветам; кривые усечения и "
                     "спектры при m=256; PCA — по 1755 цветам без центрирования "
                     "по признакам"),
        "pca_n95": {lvl: reps["fig4_manifold"][lvl]["n95"] for lvl in ("VL", "LM")},
        "pca_participation_ratio": {
            lvl: reps["fig4_manifold"][lvl]["participation_ratio"]
            for lvl in ("VL", "LM")},
        "rank_to_reach_cam16_lcd": {},
        "stress_saturation": {},
    }
    numbers["above_shelf"] = {}
    for lvl in ("VL", "LM"):
        sh = zs[f"lam0_{lvl}_{M_SPEC}"][0]
        for m_chk in (256, 1024, M_SPEC):
            lam = zs[f"lam_{lvl}_{m_chk}"]
            n_ab = int((lam > sh).sum())
            numbers["above_shelf"][f"{lvl}_m{m_chk}"] = {
                "n": n_ab,
                "lambda1_over_shelf": round(float(lam[0] / sh), 2),
                "energy_above": round(float(lam[lam > sh].sum() / lam.sum()), 4)}
    for lvl in ("VL", "LM"):
        for tag, z, src in (("separate", zs, gs), ("joint", zj, gj)):
            c = z[f"curve_{lvl}_256"]
            hit = np.argmax(c <= CAM16_LCD) + 1 if (c <= CAM16_LCD).any() else None
            numbers["rank_to_reach_cam16_lcd"][f"{lvl}_{tag}"] = int(hit) if hit else None
            ms = sorted(int(k) for k in src[lvl] if k.isdigit())
            best = min(ms, key=lambda m: src[lvl][str(m)]["stress_mean"])
            numbers["stress_saturation"][f"{lvl}_{tag}"] = {
                "m": best, "stress": round(src[lvl][str(best)]["stress_mean"], 4),
                "std": round(src[lvl][str(best)]["stress_std"], 4)}
    (A / "fig_map_structure.json").write_text(
        json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")

    print("сохранено graphics/fig_map_structure.{png,pdf}")
    print("ранг, догоняющий CAM16-LCD:", numbers["rank_to_reach_cam16_lcd"])
    print("предел качества:", numbers["stress_saturation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
