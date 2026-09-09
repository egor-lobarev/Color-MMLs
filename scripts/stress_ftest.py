#!/usr/bin/env python3
"""F-критерий значимости для всех сравнений статьи «карта vs колориметрия».

Отвечает на вопрос, который статья до сих пор обходила: значимо ли обученное
линейное отображение превосходит аналитические пространства, или разница лежит
в пределах случайности. Без этого формулировка «превосходит» не обеспечена.

Числа STRESS берутся из data/analysis/*.json (правило §6.5: руками не правятся),
критерий — utils/analyze/stress_ftest.py.

    python scripts/stress_ftest.py
    python scripts/stress_ftest.py --alpha 0.01

Результат: data/analysis/stress_ftest.json

⚠️ Две методические оговорки, обе выведены в отчёт:

1. Критерий Мелгосы выведен для STRESS, посчитанного один раз на N парах.
   Здесь для обученных карт подставляется среднее по фолдам кросс-валидации —
   разброс между фолдами в F не входит. Поэтому для карт приводится анализ
   чувствительности: как меняется вердикт при N от размера тестового фолда до
   полного набора пар.

2. На Leeds аналитические пространства оценены на всех 307 парах, а карта — на
   тестовых подмножествах при разбиении по цветовым центрам. Это сравнение на
   разных наборах пар; строгий тест требует оценки CAM16 на тех же тестовых
   парах. Пока не сделано — помечено в отчёте как approximate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.analyze.stress_ftest import stress_ftest          # noqa: E402

A = ROOT / "data" / "analysis"
OUT = A / "stress_ftest.json"

# Число пар по группам (scripts/analyze_gram_spectrum.py -> mm.build_pairs).
MUNSELL_PAIRS = {"varying-H": 1518, "varying-C": 1389, "varying-V": 1445}
MUNSELL_TOTAL = sum(MUNSELL_PAIRS.values())          # 4352

# Leeds: 307 пар всего. При разбиении по цветовым центрам «мостиковые» пары
# (концы в разных фолдах) отбрасываются, в тесте остаётся заметно меньше.
LEEDS_TOTAL = 307
LEEDS_TEST_APPROX = 198        # оценка по числу немостиковых пар


def load(name: str) -> dict:
    p = A / name
    if not p.exists():
        raise SystemExit(f"Нет {p.relative_to(ROOT)} — сначала посчитайте его скриптом.")
    return json.loads(p.read_text(encoding="utf-8"))


def gk_mean(per_group: dict) -> float:
    return sum(per_group[k] for k in ("varying-V", "varying-C", "varying-H")) / 3.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.05, help="Уровень значимости.")
    args = ap.parse_args()

    mun = load("map_munsell_groupk.json")
    leeds = load("verify_combvd_leak.json")
    joint = load("gram_spectrum_joint.json")

    results: dict[str, list] = {}

    # ---- 1. Манселл, раздельное обучение: карта против CAM16, по группам ----
    # Здесь сравнение честное: обе стороны на одних и тех же парах и группах.
    rows = []
    cam = mun["cam16_on_subset"]
    for layer in ("VL", "LM"):
        m = mun["map"][layer]["256"]["per_group"]
        for variant in ("CAM16-LCD", "CAM16-UCS", "CAM16-SCD", "CIEDE2000"):
            for g, n in MUNSELL_PAIRS.items():
                r = stress_ftest(m[g]["mean"], cam[variant][g], n_pairs=n,
                                 alpha=args.alpha,
                                 name_a=f"карта·{layer}", name_b=variant)
                rows.append({"layer": layer, "baseline": variant, "group": g,
                             **r.as_dict()})
    results["munsell_per_group"] = rows

    # ---- 2. Манселл, group-k среднее ----
    # Приближение: усреднённый по трём группам STRESS с суммарным N.
    rows = []
    for layer in ("VL", "LM"):
        s_map = mun["map"][layer]["256"]["group_k_mean"]
        for variant in ("CAM16-LCD", "CAM16-UCS", "CAM16-SCD", "CIEDE2000"):
            r = stress_ftest(s_map, gk_mean(cam[variant]), n_pairs=MUNSELL_TOTAL,
                             alpha=args.alpha,
                             name_a=f"карта·{layer}", name_b=variant)
            rows.append({"layer": layer, "baseline": variant,
                         "approximation": "group-k среднее трёх групп, N = сумма пар",
                         **r.as_dict()})
    results["munsell_group_k"] = rows

    # ---- 3. Leeds, обучение только на Leeds ----
    # Анализ чувствительности к N: от размера тестовой части до всех пар.
    rows = []
    for layer in ("VL", "LM"):
        by_center = leeds["map"][layer]["by_center"]
        best_m = min(by_center, key=lambda k: by_center[k]["mean"])
        s_map = by_center[best_m]["mean"]
        s_std = by_center[best_m]["std"]
        for variant, s_base in leeds["baselines"].items():
            for n_label, n in (("test_approx", LEEDS_TEST_APPROX),
                               ("all_pairs", LEEDS_TOTAL)):
                r = stress_ftest(s_map, s_base, n_pairs=n, alpha=args.alpha,
                                 name_a=f"карта·{layer}", name_b=variant)
                rows.append({"layer": layer, "m": best_m, "map_std": s_std,
                             "baseline": variant, "n_basis": n_label,
                             "approximation": "карта — среднее по фолдам на тестовых "
                                              "парах, бейзлайн — на всех 307",
                             **r.as_dict()})
    results["leeds_trained_on_leeds"] = rows

    # ---- 3b. Манселл, совместное обучение: по группам ----
    # Аннотация и заключение говорят о превосходстве ИМЕННО совместной карты
    # на обоих масштабах, поэтому манселловскую часть тоже надо проверить.
    rows = []
    for layer in ("VL", "LM"):
        g = joint[layer]["1024"]["groups"]
        for variant in ("CAM16-LCD", "CAM16-UCS", "CAM16-SCD", "CIEDE2000"):
            for grp, n in MUNSELL_PAIRS.items():
                r = stress_ftest(g[grp]["mean"], cam[variant][grp], n_pairs=n,
                                 alpha=args.alpha,
                                 name_a=f"совместная·{layer}", name_b=variant)
                rows.append({"layer": layer, "baseline": variant, "group": grp,
                             **r.as_dict()})
            gk_map = sum(g[k]["mean"] for k in MUNSELL_PAIRS) / 3.0
            r = stress_ftest(gk_map, gk_mean(cam[variant]), n_pairs=MUNSELL_TOTAL,
                             alpha=args.alpha,
                             name_a=f"совместная·{layer}", name_b=variant)
            rows.append({"layer": layer, "baseline": variant, "group": "group-k",
                         "approximation": "среднее трёх манселловских групп",
                         **r.as_dict()})
    results["munsell_joint"] = rows

    # ---- 4. Leeds, совместное обучение (Манселл + Leeds) ----
    rows = []
    for layer in ("VL", "LM"):
        g = joint[layer]["1024"]["groups"]["leeds"]
        for variant, s_base in leeds["baselines"].items():
            for n_label, n in (("test_approx", LEEDS_TEST_APPROX),
                               ("all_pairs", LEEDS_TOTAL)):
                r = stress_ftest(g["mean"], s_base, n_pairs=n, alpha=args.alpha,
                                 name_a=f"совместная·{layer}", name_b=variant)
                rows.append({"layer": layer, "m": "1024", "map_std": g["std"],
                             "baseline": variant, "n_basis": n_label, **r.as_dict()})
    results["leeds_joint"] = rows

    OUT.write_text(json.dumps({
        "script": "scripts/stress_ftest.py",
        "date": str(date.today()),
        "protocol": (
            "F-критерий Мелгосы (García et al., 2007): F = (STRESS_A/STRESS_B)^2, "
            f"df = N-1, двусторонний, alpha = {args.alpha}. Числа STRESS — из "
            "map_munsell_groupk.json, verify_combvd_leak.json, "
            "gram_spectrum_joint.json. Для обученных карт подставляется среднее "
            "по фолдам CV, поэтому разброс между фолдами в F не входит; там, где "
            "это существенно, приведён анализ чувствительности к N."
        ),
        "alpha": args.alpha,
        "n_pairs": {"munsell_per_group": MUNSELL_PAIRS,
                    "munsell_total": MUNSELL_TOTAL,
                    "leeds_all": LEEDS_TOTAL,
                    "leeds_test_approx": LEEDS_TEST_APPROX},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- консольная сводка ----------------
    def show(title, rows, key_fmt):
        print(f"\n=== {title} ===")
        for r in rows:
            mark = {"A": "✓ карта лучше", "B": "✗ бейзлайн лучше",
                    "ns": "— незначимо"}[r["verdict"]]
            print(f"  {key_fmt(r):46s} STRESS {r['stress_a']:.3f} vs "
                  f"{r['stress_b']:.3f}  p={r['p_value']:.4f}  {mark}")

    show("Манселл, group-k (карта m=256 против колориметрии)",
         results["munsell_group_k"],
         lambda r: f"{r['layer']} vs {r['baseline']}")
    show("Leeds, обучение только на Leeds",
         [r for r in results["leeds_trained_on_leeds"] if r["n_basis"] == "test_approx"],
         lambda r: f"{r['layer']} (m={r['m']}) vs {r['baseline']}")
    show("Манселл, совместное обучение (group-k)",
         [r for r in results["munsell_joint"] if r["group"] == "group-k"],
         lambda r: f"{r['layer']} vs {r['baseline']}")
    show("Leeds, совместное обучение",
         [r for r in results["leeds_joint"] if r["n_basis"] == "test_approx"],
         lambda r: f"{r['layer']} vs {r['baseline']}")

    ns = sum(1 for rows in results.values() for r in rows if r["verdict"] == "ns")
    tot = sum(len(rows) for rows in results.values())
    print(f"\nвсего сравнений: {tot}, из них незначимых: {ns}")
    print(f"{OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
