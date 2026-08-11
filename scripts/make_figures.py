"""
Headline figures for the Sensory Systems publication (plan doc §2, §6).
All Leeds numbers verified by scripts/verify_combvd_leak.py (Qwen-7B, by-center split).
Outputs to graphics/: fig2_main_comparison.(png|pdf), fig3_per_group_munsell.(png|pdf)

Style: minimalist — no chart titles/suptitles (descriptions live in the manuscript
captions); panel identity carried by axis labels; base font 14 pt.
The graphical abstract (fig0) is produced by scripts/make_graphical_abstract.py.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT = Path("graphics"); OUT.mkdir(exist_ok=True)
rcParams.update({
    "font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8, "figure.dpi": 140,
    "font.family": "DejaVu Sans",
})

# palette: gray = analytic colorimetry, ACCENT = MLLM mapping, green = human floor
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
    ("CIELAB (ΔE76)",       0.400, GRAY,   False),
    ("CAM16-LCD",           0.322, GRAY_D, False),
    ("CAM16-UCS",           0.287, GRAY_D, False),
    ("CAM16-SCD",           0.256, GRAY_D, False),
    ("CIEDE2000",           0.195, WARN,   False),
    ("Отобр. · LM",         0.238, ACCENT, True),
    ("Отобр. · VL",         0.227, ACCENT_D, True),
]
# Munsell (suprathreshold), group-k regime throughout (full color set for CAM16,
# object-wise CV for the mapping at m=256)
munsell = [
    ("CAM16-LCD",           0.364, GRAY_D, False),
    ("CAM16-UCS",           0.343, GRAY_D, False),
    ("Отобр. · LM",         0.173, ACCENT, True),
    ("Отобр. · VL",         0.102, ACCENT_D, True),
]


def bars(ax, data, xlabel):
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    cols = [d[2] for d in data]
    x = np.arange(len(data))
    ax.bar(x, vals, color=cols, width=0.66, zorder=3,
           edgecolor="white", linewidth=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=12, color="#222")
    ax.axhline(NOISE, ls="--", lw=1.3, color=FLOOR, zorder=2)
    xs = len(data) - 2.35                     # dash sample in the top-right free zone
    ax.plot([xs, xs + 0.35], [0.55, 0.55], ls="--", lw=1.3, color=FLOOR, clip_on=False)
    ax.text(xs + 0.45, 0.55, "пол шума 0.090", ha="left", va="center",
            color=FLOOR, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12, rotation=18, ha="right")
    ax.set_ylim(0, 0.60)
    ax.set_ylabel("STRESS  (↓ ближе к человеку)")
    ax.set_xlabel(xlabel, fontsize=14, labelpad=8)
    ax.grid(axis="y", color="#EEE", zorder=0)


# ============================================================== Fig 2: comparison
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"width_ratios": [5, 3]})
bars(a1, leeds, "Пороговые различия · COMBVD-Leeds")
bars(a2, munsell, "Надпороговые различия · Манселл (group-k)")
from matplotlib.patches import Patch
leg = [Patch(fc=ACCENT_D, label="Линейное отображение представлений МБЯМ (Qwen2.5-VL-7B)"),
       Patch(fc=GRAY_D, label="Аналитическая колориметрия"),
       Patch(fc=WARN, label="CIEDE2000 (спец. формула малых различий)")]
fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.04), fontsize=12)
fig.tight_layout(rect=(0, 0.03, 1, 1.0))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig2_main_comparison.{ext}", bbox_inches="tight")
plt.close(fig)


# ============================================ Fig 3: per-group H/C/V (mapping vs CAM16)
# verified by scripts/map_munsell_group_stress.py (1755 colors, object-wise 5-fold,
# per-group k). CAM16-LCD recomputed on the same color subset.
cats = ["Тон\n(varying-H)", "Насыщенность\n(varying-C)", "Светлота\n(varying-V)", "Среднее\n(Group-k)"]
cam16   = [0.504, 0.286, 0.074, 0.288]  # delta_E-proper (map_munsell_group_stress.py)
map_vl  = [0.110, 0.120, 0.076, 0.102]
map_lm  = [0.230, 0.192, 0.096, 0.173]
err_vl  = [0.008, 0.015, 0.006, 0.0]
err_lm  = [0.021, 0.026, 0.011, 0.0]

fig, ax = plt.subplots(figsize=(10.5, 5.4))
x = np.arange(len(cats)); w = 0.26
b0 = ax.bar(x - w, cam16, w, label="CAM16-LCD", color=GRAY_D, zorder=3)
b1 = ax.bar(x,      map_vl, w, yerr=err_vl, capsize=3, ecolor="#333",
            label="Отображение · VL (m=256)", color=ACCENT_D, zorder=3)
b2 = ax.bar(x + w,  map_lm, w, yerr=err_lm, capsize=3, ecolor="#333",
            label="Отображение · LM (m=256)", color=ACCENT, zorder=3)
for bset in (b0, b1, b2):
    for r in bset:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=11, color="#333")
# highlight the hue win
ax.annotate("×4.5", xy=(0 - w, 0.504), xytext=(0, 0.47), fontsize=14,
            color=ACCENT_D, fontweight="bold", ha="center")
ax.axvline(2.5, color="#DDD", lw=1, zorder=1)  # separates the Group-k mean
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=13)
ax.set_ylabel("STRESS  (↓ ближе к человеку)")
ax.set_ylim(0, 0.60)
ax.grid(axis="y", color="#EEE", zorder=0)
ax.legend(frameon=False, fontsize=13, loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig3_per_group_munsell.{ext}", bbox_inches="tight")
plt.close(fig)

print("saved:", *[p.name for p in sorted(OUT.glob('fig[23]*'))])
