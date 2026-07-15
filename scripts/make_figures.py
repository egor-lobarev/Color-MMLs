"""
Headline figures for the Sensory Systems publication (plan doc §2, §6).
All Leeds numbers verified by scripts/verify_combvd_leak.py (Qwen-7B, by-center split).
Outputs to graphics/: fig0_graphical_abstract.(png|pdf), fig2_main_comparison.(png|pdf)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT = Path("graphics"); OUT.mkdir(exist_ok=True)
rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 140,
    "font.family": "DejaVu Sans",
})

# palette: gray = analytic colorimetry, ACCENT = MLLM map, green = human floor
ACCENT = "#3B6FB5"      # MLLM
ACCENT_D = "#26456f"
GRAY = "#9AA0A6"
GRAY_D = "#5F6368"
FLOOR = "#2E9E6B"
WARN = "#C77B30"

NOISE = 0.090

# --- verified numbers (STRESS, lower is better) ------------------------------
# All CAM16 values are formula-proper delta_E (K_L applied); sources:
# verify_combvd_leak.py (Leeds, by-center split for MLLM) and
# cam16_munsell_group_stress.py / map_munsell_group_stress.py (Munsell, group-k).
leeds = [
    ("CIELAB\n(ΔE76)",      0.400, GRAY,   False),
    ("CAM16-LCD",           0.322, GRAY_D, False),
    ("CAM16-UCS",           0.287, GRAY_D, False),
    ("CAM16-SCD",           0.256, GRAY_D, False),
    ("CIEDE2000",           0.195, WARN,   False),
    ("МЯМ · LM",            0.238, ACCENT, True),
    ("МЯМ · VL",            0.227, ACCENT_D, True),
]
# Munsell (suprathreshold), group-k regime throughout (full color set for CAM16,
# object-wise CV for the map at m=256)
munsell = [
    ("CAM16-LCD",           0.364, GRAY_D, False),
    ("CAM16-UCS",           0.343, GRAY_D, False),
    ("МЯМ · LM",            0.173, ACCENT, True),
    ("МЯМ · VL",            0.102, ACCENT_D, True),
]


def bars(ax, data, title):
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    cols = [d[2] for d in data]
    x = np.arange(len(data))
    b = ax.bar(x, vals, color=cols, width=0.66, zorder=3,
               edgecolor="white", linewidth=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9.5, color="#222")
    ax.axhline(NOISE, ls="--", lw=1.3, color=FLOOR, zorder=2)
    ax.text(len(data) - 0.4, NOISE + 0.006, "пол шума человека 0.090",
            ha="right", va="bottom", color=FLOOR, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 0.60)
    ax.set_ylabel("STRESS  (↓ ближе к человеку)")
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", color="#EEE", zorder=0)
    return b


# ============================================================== Fig 2: comparison
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"width_ratios": [5, 3]})
bars(a1, leeds, "Пороговые различия · COMBVD-Leeds")
bars(a2, munsell, "Надпороговые различия · Манселл (group-k)")
fig.suptitle("Согласованность метрики с психофизикой цветовых различий человека",
             fontsize=13, y=1.00)
# legend
from matplotlib.patches import Patch
leg = [Patch(fc=ACCENT_D, label="Карта из эмбеддингов МЯМ (Qwen2.5-VL-7B)"),
       Patch(fc=GRAY_D, label="Аналитическая колориметрия"),
       Patch(fc=WARN, label="CIEDE2000 (спец. формула малых различий)")]
fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.04), fontsize=9)
fig.tight_layout(rect=(0, 0.02, 1, 0.99))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig2_main_comparison.{ext}", bbox_inches="tight")
plt.close(fig)


# ==================================================== Fig 0: graphical abstract
# Left: method pipeline. Right: two STRESS number lines (suprathreshold Munsell,
# threshold Leeds) on a shared scale. All values delta_E-proper / by-center split.
fig = plt.figure(figsize=(11.6, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.9], hspace=1.15,
                      left=0.02, right=0.97, top=0.80, bottom=0.17)

# ---- left: pipeline schematic ----
axp = fig.add_subplot(gs[:, 0]); axp.axis("off")
axp.set_xlim(0, 1); axp.set_ylim(0, 1)
boxes = [
    (0.90, "Однородные цветовые патчи\nМанселл · COMBVD", "#F2F2F2", "#666"),
    (0.62, "МЯМ (Qwen2.5-VL)\nэмбеддинги e_VL, e_LM", "#E8EEF7", ACCENT_D),
    (0.34, "Линейная карта A\n(метрическое обучение\nна психофизике человека)", "#E8EEF7", ACCENT_D),
    (0.06, "‖A·Δe‖ ≈ воспринимаемое\nцветовое различие", "#EAF5EF", FLOOR),
]
for y, txt, fc, ec in boxes:
    axp.text(0.5, y, txt, ha="center", va="center", fontsize=9.3, color="#222",
             bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec=ec, lw=1.2))
for y0, y1 in [(0.83, 0.72), (0.55, 0.46), (0.24, 0.15)]:
    axp.annotate("", xy=(0.5, y1), xytext=(0.5, y0),
                 arrowprops=dict(arrowstyle="->", color="#888", lw=1.4))

# ---- right: two number lines on a shared STRESS scale ----
XLIM = (0.05, 0.44)

def numline(ax, pts, title):
    ax.set_xlim(*XLIM); ax.set_ylim(-1.15, 1.15)
    ax.get_yaxis().set_visible(False)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.tick_params(labelsize=8.5)
    for val, lab, col, mark, filled, side in pts:
        ax.plot([val], [0], marker=mark, ms=15 if mark == "|" else 11,
                mfc=col if filled else "white", mec=col, mew=2.2, zorder=4,
                clip_on=False)
        if lab:
            ax.annotate(lab, xy=(val, 0), xytext=(val, side * 0.80),
                        ha="center", va="center", fontsize=8.6, color=col,
                        fontweight="bold" if filled and mark == "o" else "normal",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.9))
    ax.set_title(title, fontsize=10.5, loc="left", pad=4)

ax1 = fig.add_subplot(gs[0, 1])
numline(ax1, [
    (0.102, "карта МЯМ  0.102",           ACCENT_D, "o", True,  1),
    (0.288, None,                          GRAY_D,   "o", False, -1),
    (0.297, "семейство CAM16\nLCD · UCS · SCD", GRAY_D, "o", False, -1),
    (0.303, None,                          GRAY_D,   "o", False, -1),
], "Надпороговые различия · Манселл (group-k)")
ax1.annotate("", xy=(0.115, 0.42), xytext=(0.283, 0.42),
             arrowprops=dict(arrowstyle="->", color=ACCENT_D, lw=1.5))
ax1.text(0.198, 0.68, "почти ×3", ha="center", color=ACCENT_D, fontsize=8.6)

ax2 = fig.add_subplot(gs[1, 1])
numline(ax2, [
    (NOISE, "пол шума\nчеловека",          FLOOR,    "|", True,  1),
    (0.195, "CIEDE2000",                   WARN,     "o", False, -1),
    (0.227, "карта МЯМ  0.227",            ACCENT_D, "o", True,  1),
    (0.256, None,                          GRAY_D,   "o", False, -1),
    (0.287, "семейство CAM16\nSCD · UCS · LCD", GRAY_D, "o", False, -1),
    (0.322, None,                          GRAY_D,   "o", False, -1),
], "Пороговые различия · COMBVD-Leeds (сплит по центрам)")
fig.text(0.62, 0.015, "STRESS · рассогласование с психофизикой цветоразличения  (↓ лучше)",
         ha="center", fontsize=10)

fig.suptitle("Внутренние представления МЯМ воспроизводят метрику цветовых различий человека:\n"
             "линейная карта превосходит каждый вариант CAM16 на его масштабе — модель не обучалась цветоразличению",
             fontsize=12, y=0.97)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig0_graphical_abstract.{ext}", bbox_inches="tight")
plt.close(fig)

# ============================================ Fig 3: per-group H/C/V (map vs CAM16)
# verified by scripts/map_munsell_group_stress.py (1755 colors, object-wise 5-fold,
# per-group k). CAM16-LCD recomputed on the same color subset.
cats = ["Тон\n(varying-H)", "Насыщенность\n(varying-C)", "Светлота\n(varying-V)", "Среднее\n(Group-k)"]
cam16   = [0.504, 0.286, 0.074, 0.288]  # delta_E-proper (map_munsell_group_stress.py)
map_vl  = [0.110, 0.120, 0.076, 0.102]
map_lm  = [0.230, 0.192, 0.096, 0.173]
err_vl  = [0.008, 0.015, 0.006, 0.0]
err_lm  = [0.021, 0.026, 0.011, 0.0]

fig, ax = plt.subplots(figsize=(9.4, 4.8))
x = np.arange(len(cats)); w = 0.26
b0 = ax.bar(x - w, cam16, w, label="CAM16-LCD", color=GRAY_D, zorder=3)
b1 = ax.bar(x,      map_vl, w, yerr=err_vl, capsize=3, ecolor="#333",
            label="Карта · VL (m=256)", color=ACCENT_D, zorder=3)
b2 = ax.bar(x + w,  map_lm, w, yerr=err_lm, capsize=3, ecolor="#333",
            label="Карта · LM (m=256)", color=ACCENT, zorder=3)
for bars in (b0, b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8.3, color="#333")
# highlight the hue win
ax.annotate("×4.5", xy=(0 - w, 0.504), xytext=(0, 0.47), fontsize=10,
            color=ACCENT_D, fontweight="bold", ha="center")
ax.axvline(2.5, color="#DDD", lw=1, zorder=1)  # separates the Group-k mean
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9.5)
ax.set_ylabel("STRESS  (↓ ближе к человеку)")
ax.set_ylim(0, 0.60)
ax.set_title("Где карта из эмбеддингов обгоняет CAM16 на цепочках Манселла:\n"
             "выигрыш сосредоточен в тоне и насыщенности, по светлоте — паритет",
             fontsize=12.5, pad=10)
ax.grid(axis="y", color="#EEE", zorder=0)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig3_per_group_munsell.{ext}", bbox_inches="tight")
plt.close(fig)

print("saved:", *[p.name for p in sorted(OUT.glob('fig*'))])
