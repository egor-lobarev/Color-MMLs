"""
Собирает ЕДИНЫЙ РЕЕСТР всех чисел статьи «Сенсорные системы» из JSON-файлов
data/analysis/*.json в человекочитаемый манифест data/analysis/ARTICLE_NUMBERS.md.

Правило прозрачности: каждое число в тексте статьи должно быть выведено из
файла data/analysis/*.json, который порождается конкретным скриптом. Если файл
отсутствует, манифест помечает число как НЕ ПОСЧИТАНО (запустите скрипт из
колонки «Источник»). Числа, взятые из ВКР и пока не пересчитанные локально,
собраны в отдельной секции с пометкой.

Запуск:  python scripts/collect_article_numbers.py
"""
import json
from datetime import date
from pathlib import Path

A = Path("data/analysis")
OUT = A / "ARTICLE_NUMBERS.md"

# число из ВКР, не пересчитывалось локально (нет сырых повторных измерений)
NOISE_FLOOR = 0.0896


def load(name):
    p = A / name
    if not p.exists():
        return None
    return json.load(open(p))


def fmt(x, digits=3):
    if x is None:
        return "⚠️ не посчитано"
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else str(x)


def ms(d, digits=3):
    """dict {mean, std} -> 'mean ± std'."""
    if d is None:
        return "⚠️ не посчитано"
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def main():
    cam16 = load("cam16_munsell.json")
    mapg = load("map_munsell_groupk.json")
    leak = load("verify_combvd_leak.json")
    gram = load("gram_spectrum.json")
    joint = load("gram_spectrum_joint.json")
    cross = load("cross_dataset.json")
    reps = load("representations_fig4_fig5.json")
    tsz = load("testsize_fig6_fig7.json")
    dec = load("decoded_coords_control.json")
    dE7 = load("fig7_error_dE.json")
    fig0 = load("fig0_abstract.json")

    L = []
    L.append("# Реестр чисел статьи «Сенсорные системы»\n")
    L.append(f"*Собрано {date.today()} скриптом `scripts/collect_article_numbers.py`. "
             "Каждое число прослеживается до JSON-файла в `data/analysis/` и "
             "скрипта, который его создал. Правка чисел руками запрещена — "
             "перезапустите скрипт-источник.*\n")

    # -------- Табл. 1: CAM16 на полном Манселле --------
    L.append("## Табл. 1 статьи — CAM16 на Манселле (global-k vs group-k)")
    L.append("Источник: `cam16_munsell.json` ← `scripts/cam16_munsell_group_stress.py`\n")
    if cam16:
        L.append("| Пространство | varying-V | varying-C | varying-H | Group-k | Global-k |")
        L.append("|---|---|---|---|---|---|")
        for name, r in cam16["results"].items():
            pg = r["per_group"]
            L.append(f"| {name} | {pg['varying-V']:.3f} | {pg['varying-C']:.3f} | "
                     f"{pg['varying-H']:.3f} | {r['group_k_mean']:.3f} | {r['global_k']:.3f} |")
    else:
        L.append("⚠️ не посчитано — запустите скрипт")
    L.append("")

    # -------- §3.3 / рис. 2, 3: карта vs CAM16 на подмножестве --------
    L.append("## §3.3, рис. 2–3 — карта vs CAM16, подмножество 1755 цветов (group-k)")
    L.append("Источник: `map_munsell_groupk.json` ← `scripts/map_munsell_group_stress.py`\n")
    if mapg:
        cam = mapg["cam16_on_subset"]
        L.append("| Метод | varying-V | varying-C | varying-H | Group-k |")
        L.append("|---|---|---|---|---|")
        for v, pg in cam.items():
            mean = (pg["varying-V"] + pg["varying-C"] + pg["varying-H"]) / 3
            L.append(f"| {v} | {pg['varying-V']:.3f} | {pg['varying-C']:.3f} | "
                     f"{pg['varying-H']:.3f} | {mean:.3f} |")
        for layer in ("VL", "LM"):
            for m, r in mapg["map"][layer].items():
                pg = r["per_group"]
                L.append(f"| Карта {layer} (m={m}) | {pg['varying-V']['mean']:.3f} | "
                         f"{pg['varying-C']['mean']:.3f} | {pg['varying-H']['mean']:.3f} | "
                         f"{r['group_k_mean']:.3f} |")
    else:
        L.append("⚠️ не посчитано")
    L.append("")

    # -------- §3.4: Leeds, бейзлайны и проверка утечки --------
    L.append("## §3.4 — пороговые различия Leeds: бейзлайны и карта (сплит по центрам)")
    L.append("Источник: `verify_combvd_leak.json` ← `scripts/verify_combvd_leak.py`\n")
    if leak:
        L.append("| Бейзлайн | STRESS |")
        L.append("|---|---|")
        for k, v in leak["baselines"].items():
            L.append(f"| {k} | {fmt(v)} |")
        L.append(f"| Шум повторных измерений (66 пар, из ВКР) | {leak['noise_floor_repeats']:.3f} |")
        L.append("")
        L.append("| Слой | Сплит | m | STRESS |")
        L.append("|---|---|---|---|")
        for layer in ("VL", "LM"):
            for proto, tag in [("by_center", "по центрам (корректно)"),
                               ("by_pair_leaky", "по парам (утечка, как ВКР)")]:
                for m, r in leak["map"][layer][proto].items():
                    L.append(f"| {layer} | {tag} | {m} | {ms(r)} |")
    else:
        L.append("⚠️ не посчитано")
    L.append("")

    # -------- §3.5 / рис. 8 --------
    L.append("## §3.5, рис. 8 — спектр Грама, только Манселл (свип m)")
    L.append("Источник: `gram_spectrum.json` (+`.npz`) ← "
             "`scripts/analyze_gram_spectrum.py`\n")
    if gram:
        L.append("| m | LM: STRESS | LM: r_q | VL: STRESS | VL: r_q |")
        L.append("|---|---|---|---|---|")
        for m in gram["LM"]:
            a, b = gram["LM"][m], gram["VL"][m]
            L.append(f"| {m} | {a['stress_mean']:.3f} ± {a['stress_std']:.3f} | "
                     f"{a['rq_mean']:.0f} | {b['stress_mean']:.3f} ± "
                     f"{b['stress_std']:.3f} | {b['rq_mean']:.0f} |")
    else:
        L.append("⚠️ не посчитано")
    L.append("")

    # -------- §3.6 / рис. 9 + табл. 2 --------
    L.append("## §3.6, рис. 9, табл. 2 — совместное обучение Манселл + Leeds")
    L.append("Источник: `gram_spectrum_joint.json` ← "
             "`scripts/analyze_gram_spectrum.py --joint`\n")
    if joint:
        L.append("| m | LM: Group STRESS | LM: Leeds | LM: r_q | VL: Group STRESS | VL: Leeds | VL: r_q |")
        L.append("|---|---|---|---|---|---|---|")
        for m in joint["LM"]:
            a, b = joint["LM"][m], joint["VL"][m]
            L.append(f"| {m} | {a['stress_mean']:.3f} ± {a['stress_std']:.3f} | "
                     f"{ms(a['groups']['leeds'])} | {a['rq_mean']:.0f} | "
                     f"{b['stress_mean']:.3f} ± {b['stress_std']:.3f} | "
                     f"{ms(b['groups']['leeds'])} | {b['rq_mean']:.0f} |")
    else:
        L.append("⚠️ не посчитано")
    L.append("")

    L.append("### Кросс-датасет (табл. 2, ячейки †)")
    L.append("Источник: `cross_dataset.json` ← `scripts/analyze_gram_spectrum.py --cross`\n")
    if cross:
        L.append("| Слой | Манселл→Leeds | Leeds→Манселл (group-k) | …по осям H/C/V |")
        L.append("|---|---|---|---|")
        for layer, r in cross.items():
            g = r["leeds_to_munsell_groups"]
            L.append(f"| {layer} | {r['munsell_to_leeds']:.3f} | "
                     f"{r['leeds_to_munsell_groupk']:.3f} | "
                     f"{g['varying-H']:.3f} / {g['varying-C']:.3f} / {g['varying-V']:.3f} |")
    else:
        L.append("⚠️ не посчитано")
    L.append("")

    # -------- §3.7 / рис. 4, 5 --------
    L.append("## §3.7, рис. 4–5 — манифолд и структура карты A")
    L.append("Источник: `representations_fig4_fig5.json` ← "
             "`scripts/analyze_representations.py`\n")
    if reps:
        f4 = reps["fig4_manifold"]
        L.append("| Слой | PR | n95 PCA | MLE (k=20) | TwoNN |")
        L.append("|---|---|---|---|---|")
        for n, r in f4.items():
            L.append(f"| {n} | {r['participation_ratio']} | {r['n95']} | "
                     f"{r['mle_k']['20']} | {r['twonn']} |")
        f5 = reps["fig5_map_spectrum"]
        L.append("")
        for n, r in f5.items():
            sv = r["stress_vs_rank"]
            L.append(f"- {n}: 95% энергии A(m=256) в {r['n95_energy_of_256']} сингулярных "
                     f"числах; STRESS при r=8: {sv['8']:.3f}, r=256: {sv['256']:.3f}")
    else:
        L.append("⚠️ не посчитано")
    L.append("")

    # -------- §3.8 / рис. 6 --------
    L.append("## §3.8, рис. 6 — зависимость от объёма данных")
    L.append("Источник: `testsize_fig6_fig7.json` ← `scripts/analyze_testsize_errors.py`\n")
    if tsz:
        f6 = tsz["fig6_testsize"]
        L.append("| Датасет | Слой | " + " | ".join(f"f={f}" for f in f6["fracs"]) + " |")
        L.append("|---|---|" + "---|" * len(f6["fracs"]))
        for ds in ("munsell", "leeds"):
            for n, rr in f6[ds].items():
                cells = []
                for f in f6["fracs"]:
                    r = rr[str(f)]
                    cells.append("—" if r["mean"] != r["mean"] else f"{r['mean']:.3f}")
                L.append(f"| {ds} | {n} | " + " | ".join(cells) + " |")
    else:
        L.append("⚠️ не посчитано")
    L.append("")

    # -------- §3.9 / рис. 7 --------
    L.append("## §3.9, рис. 7 — ошибки по цветовому телу")
    L.append("Источники: `fig7_error_dE.json`, `testsize_fig6_fig7.json`, "
             "`decoded_coords_control.json`\n")
    if dE7:
        for n, r in dE7["results"].items():
            L.append(f"- ΔE-версия, {n}: среднее {r['mean_dE']} ΔE, "
                     f"максимум ячейки {r['max_cell_dE']} ΔE")
    if tsz:
        for n, r in tsz["fig7_error_metric"].items():
            L.append(f"- безразмерная версия, {n}: средний |k·d̂−1| = "
                     f"{r['mean_step_err']}, максимум {r['max_cell']}")
    if dec:
        for n, r in dec["results"].items():
            L.append(f"- контроль (декодированные координаты), {n}: group-k "
                     f"{r['group_k_mean']:.3f}")
    if not (dE7 or tsz or dec):
        L.append("⚠️ не посчитано")
    L.append("")

    # -------- рис. 1 (графический абстракт) --------
    L.append("## Рис. 1 — графический абстракт (зум-пары)")
    L.append("Источник: `fig0_abstract.json` ← `scripts/make_graphical_abstract.py` "
             "(гистограммы — из `map_munsell_groupk.json` и `verify_combvd_leak.json`)\n")
    if fig0:
        for g, z in fig0["zoom_pairs"].items():
            L.append(f"- {g}: {z['munsell_i']} ↔ {z['munsell_j']} — CAM16-UCS "
                     f"{z['ratio_cam']:.2f}, карта {z['ratio_map']:.2f} "
                     f"(медианный |err| по группе: CAM16 {z['median_abs_err_cam']:.2f}, "
                     f"карта {z['median_abs_err_map']:.2f})")
    else:
        L.append("⚠️ не посчитано")
    L.append("")

    # -------- константы и числа из ВКР --------
    L.append("## Константы и числа из ВКР (локально пока не пересчитаны)")
    L.append(f"- Пол шума повторных измерений Leeds (66 пар): **{NOISE_FLOOR}** — из ВКР; "
             "сырые повторные измерения в репо отсутствуют.")
    L.append("- Исходная геометрия до обучения (Т1: 0.491/0.465 Манселл, 0.396/0.448 Leeds) — ВКР.")
    L.append("- Линейное декодирование координат: R²>0.998 (LM 0.9996/0.90 ΔE, VL 0.9989/1.44 ΔE) — ВКР; "
             "ΔE-часть локально воспроизведена в `fig7_error_dE.json` (0.55/0.70).")
    L.append("- Дипломные joint-числа (STRESS 0.25–0.27, m_opt 14/32) — ВКР; наш пересчёт "
             "с reg=0 в `gram_spectrum_joint.json` (см. §5.1.3 плана об артефакте reg×m).")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"saved {OUT}")
    missing = [n for n, d in [("cam16_munsell.json", cam16),
                              ("map_munsell_groupk.json", mapg),
                              ("verify_combvd_leak.json", leak),
                              ("gram_spectrum.json", gram),
                              ("gram_spectrum_joint.json", joint),
                              ("cross_dataset.json", cross),
                              ("representations_fig4_fig5.json", reps),
                              ("testsize_fig6_fig7.json", tsz),
                              ("decoded_coords_control.json", dec),
                              ("fig7_error_dE.json", dE7),
                              ("fig0_abstract.json", fig0)] if d is None]
    print("missing:", missing if missing else "none — все источники на месте")


if __name__ == "__main__":
    main()
