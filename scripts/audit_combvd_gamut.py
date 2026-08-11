#!/usr/bin/env python3
"""Аудит COMBVD: гамут sRGB и разрешающая способность 8-битного носителя.

Обе потери возникают в scripts/generate_combvd_pictures.py при рендере патча:

    XYZ (float64)
      -> lin_rgb = XYZ @ M                     линейный sRGB
      -> np.clip(lin_rgb, 0, 1)      строка 55  ПОТЕРЯ 1: клиппинг гамута
      -> гамма-кодирование sRGB      строки 59-63
      -> np.round(srgb * 255)        строка 136 ПОТЕРЯ 2: квантование в 8 бит
      -> Image.new("RGB", ...)       строка 137
      -> PNG (без потерь)

Вне-гамутные цвета клиппятся, из-за чего расстояние в паре искажается
систематически (сильнее всего страдают насыщенные цвета). Пары, различающиеся
менее чем на --min-drgb8 кодовых значений, неразличимы на носителе: модель
физически не может извлечь из них сигнал. И те, и другие исключаются
из анализа и из извлечения эмбеддингов.

Скрипт — единственный источник правды о том, какие пары и цвета идут в работу:

    data/analysis/combvd_audit.json   числа для статьи (протокол §6.5)
    data/combvd_pairs_clean.csv       отфильтрованная таблица пар
    data/colors/combvd/keep_colors.json  индексы цветов, нужные для эмбеддингов

Запуск (CPU, ~10 c):

    python scripts/audit_combvd_gamut.py
    python scripts/audit_combvd_gamut.py --min-drgb8 2   # + отсев по разрешению
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "combvd_pairs.csv"
COLORS = ROOT / "data" / "colors" / "combvd"
OUT_JSON = ROOT / "data" / "analysis" / "combvd_audit.json"
OUT_PAIRS = ROOT / "data" / "combvd_pairs_clean.csv"
OUT_KEEP = COLORS / "keep_colors.json"

# bfd_p-c исключён целиком: медиана различия пар ΔE00 = 0.42, то есть набор почти
# полностью лежит ниже разрешения 8-битного носителя (после фильтров осталось бы
# 76 пар из 200). Статистика по нему всё равно считается и пишется в отчёт —
# ею обосновывается исключение в Методах.
EXCLUDED = ("bfd_p-c",)

# XYZ -> линейный sRGB (та же матрица, что в scripts/generate_combvd_pictures.py)
XYZ_TO_LINRGB = np.array(
    (
        (3.2404542, -0.9692660, 0.0556434),
        (-1.5371385, 1.8760108, -0.2040259),
        (-0.4985314, 0.0415560, 1.0572252),
    ),
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=None,
                   help=f"Подмножество датасетов. По умолчанию все, кроме "
                        f"исключённых: {', '.join(EXCLUDED)}.")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Допуск при проверке принадлежности гамуту.")
    p.add_argument("--min-drgb8", type=int, default=2,
                   help="Минимальное расстояние пары в 8-битных кодовых значениях "
                        "(0 = не фильтровать). По умолчанию 2: при разнице в одно "
                        "кодовое значение ошибка квантования ±0.5 к.з. составляет "
                        "±50%% от самой разницы, при 0 картинки физически идентичны.")
    return p.parse_args()


def load_colors(ds: str) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (XYZ до клиппинга, 8-битные RGB) для датасета."""
    man = json.loads((COLORS / ds / "manifest.json").read_text(encoding="utf-8"))
    items = sorted(man["items"], key=lambda it: int(it["index"]))
    xyz = np.array([it["xyz"] for it in items], dtype=np.float64)
    rgb8 = np.array([it["rgb_255"] for it in items], dtype=np.float64)
    return xyz, rgb8


def main() -> int:
    args = parse_args()
    pairs = pd.read_csv(PAIRS)
    all_names = sorted(pairs["dataset"].unique())
    names = args.datasets or [d for d in all_names if d not in EXCLUDED]
    # исключённые считаем тоже — но только ради отчёта, в рабочий набор не идут
    audited = list(dict.fromkeys(list(names) + [d for d in all_names if d in EXCLUDED]))

    report: dict[str, dict] = {}
    keep_pairs, keep_colors = [], {}

    for ds in audited:
        g = pairs[pairs["dataset"] == ds].reset_index(drop=True)
        xyz, rgb8 = load_colors(ds)

        # вне гамута: линейный sRGB выходит за [0, 1] хотя бы по одному каналу
        lin = xyz @ XYZ_TO_LINRGB
        oog = ((lin < -args.tol) | (lin > 1.0 + args.tol)).any(axis=1)

        i, j = g["i"].to_numpy(), g["j"].to_numpy()
        pair_oog = oog[i] | oog[j]
        # расстояние пары в кодовых значениях (макс. по каналу)
        drgb8 = np.abs(rgb8[i] - rgb8[j]).max(axis=1)

        keep = ~pair_oog
        if args.min_drgb8 > 0:
            keep &= drgb8 >= args.min_drgb8

        kept = g[keep].copy()
        kept["drgb8"] = drgb8[keep].astype(int)

        # цвета, оставшиеся хотя бы в одной паре — только их и нужно прогонять
        used = np.unique(np.concatenate([kept["i"].to_numpy(), kept["j"].to_numpy()])) \
            if len(kept) else np.array([], dtype=int)

        if ds not in EXCLUDED:
            keep_pairs.append(kept)
            keep_colors[ds] = used.astype(int).tolist()

        report[ds] = {
            "excluded": ds in EXCLUDED,
            "colors_total": int(len(xyz)),
            "colors_out_of_gamut": int(oog.sum()),
            "colors_kept": int(len(used)),
            "pairs_total": int(len(g)),
            "pairs_dropped_gamut": int(pair_oog.sum()),
            "pairs_dropped_resolution": int((~pair_oog & (drgb8 < args.min_drgb8)).sum())
            if args.min_drgb8 > 0 else 0,
            "pairs_kept": int(keep.sum()),
            "drgb8": {
                "collapsed_identical": int((drgb8 == 0).sum(),),
                "le_1": int((drgb8 <= 1).sum()),
                "median": float(np.median(drgb8)),
                "p90": float(np.percentile(drgb8, 90)),
                "max": float(drgb8.max()),
            },
            "dv": {
                "min": float(g["dv"].min()), "max": float(g["dv"].max()),
                "median": float(g["dv"].median()), "unique": int(g["dv"].round(4).nunique()),
            },
        }

    clean = pd.concat(keep_pairs, ignore_index=True)
    clean.to_csv(OUT_PAIRS, index=False)
    OUT_KEEP.write_text(json.dumps(keep_colors, indent=2), encoding="utf-8")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "script": "scripts/audit_combvd_gamut.py",
        "date": str(date.today()),
        "protocol": (
            "Вне гамута = линейный sRGB вне [0,1] хотя бы по одному каналу "
            f"(допуск {args.tol}); XYZ берётся из manifest.json до клиппинга. "
            "Пара исключается, если вне гамута хотя бы один из двух цветов. "
            f"Порог разрешения: drgb8 >= {args.min_drgb8} "
            "(drgb8 = макс. по каналу разница 8-битных кодовых значений)."
        ),
        "min_drgb8": args.min_drgb8,
        "excluded_datasets": {
            ds: "медиана различия пар ΔE00 = 0.42 — набор почти целиком ниже "
                "разрешения 8-битного носителя"
            for ds in EXCLUDED if ds in report
        },
        "totals": {
            "pairs_total_in_source": int(len(pairs)),
            "pairs_kept": int(len(clean)),
            "colors_kept": int(sum(len(v) for v in keep_colors.values())),
            "datasets_used": names,
        },
        "datasets": report,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- консольная сводка ----
    hdr = f"{'датасет':12s} {'пар':>6s} {'-гамут':>7s} {'-разр.':>7s} {'осталось':>9s} {'цветов':>7s}"
    print(hdr); print("-" * len(hdr))
    for ds in names:
        r = report[ds]
        print(f"{ds:12s} {r['pairs_total']:6d} {r['pairs_dropped_gamut']:7d} "
              f"{r['pairs_dropped_resolution']:7d} {r['pairs_kept']:9d} {r['colors_kept']:7d}")
    print("-" * len(hdr))
    print(f"{'ИТОГО':12s} {sum(report[d]['pairs_total'] for d in names):6d} {'':7s} {'':7s} "
          f"{len(clean):9d} {sum(len(v) for v in keep_colors.values()):7d}")
    for ds in EXCLUDED:
        if ds in report:
            r = report[ds]
            print(f"\nисключён: {ds} ({r['pairs_total']} пар; после фильтров осталось бы "
                  f"{r['pairs_kept']}) — ниже разрешения носителя")
    print(f"\n{OUT_PAIRS.relative_to(ROOT)}\n{OUT_KEEP.relative_to(ROOT)}\n"
          f"{OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
