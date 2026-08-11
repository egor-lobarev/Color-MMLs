#!/usr/bin/env python3
"""Извлечение эмбеддингов COMBVD для одной модели.

Модели запускаются по отдельности (каждой нужен свой venv), поэтому скрипт
рассчитан на один прогон = одна модель. Извлекаются только цвета, оставшиеся
после аудита гамута (scripts/audit_combvd_gamut.py): вне-гамутные клиппятся при
рендере и в анализ не идут.

Прогон возобновляемый: уже посчитанные цвета пропускаются, поэтому после падения
или таймаута достаточно перезапустить ту же команду.

    python scripts/extract_combvd_embeddings.py --config configs/combvd_qwen_2.5_7B.json
    python scripts/extract_combvd_embeddings.py --config <cfg> --datasets witt rit-dupont
    python scripts/extract_combvd_embeddings.py --config <cfg> --dry-run

Подробности запуска на сервере — README_SERVER_COMBVD.md
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

# torch/transformers подтягиваются только при реальном прогоне: --dry-run должен
# работать на машине без GPU-окружения, чтобы план проверялся до запуска

ROOT = Path(__file__).resolve().parent.parent
KEEP_COLORS = ROOT / "data" / "colors" / "combvd" / "keep_colors.json"

# файлы, по наличию которых цвет считается посчитанным
REQUIRED = ("vision_pooled_mean.npy", "projected_pooled_mean.npy", "lm_pooled_mean.npy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="JSON-конфиг модели.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Подмножество датасетов (по умолчанию — из конфига).")
    p.add_argument("--device", default=None, help="Переопределить device из конфига.")
    p.add_argument("--limit", type=int, default=None,
                   help="Обработать не более N цветов на датасет (для проверки).")
    p.add_argument("--dry-run", action="store_true",
                   help="Показать план работ и выйти, модель не загружается.")
    p.add_argument("--overwrite", action="store_true",
                   help="Пересчитать даже уже посчитанные цвета.")
    return p.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Конфиг не найден: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_keep_colors() -> Dict[str, List[int]]:
    if not KEEP_COLORS.exists():
        raise SystemExit(
            f"Не найден {KEEP_COLORS.relative_to(ROOT)}.\n"
            "Сначала выполните: python scripts/audit_combvd_gamut.py"
        )
    return {k: list(v) for k, v in json.loads(KEEP_COLORS.read_text()).items()}


def is_done(color_dir: Path) -> bool:
    return all((color_dir / f).exists() for f in REQUIRED)


def color_meta(dataset_dir: Path) -> Dict[int, dict]:
    """index -> метаданные цвета из manifest.json датасета."""
    man = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return {int(it["index"]): it for it in man.get("items", [])}


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_root = ROOT / cfg.get("dataset_root", "data/colors/combvd")
    out_root = ROOT / cfg["outdir_root"]
    model_name = cfg["model"]
    prompt = cfg.get("prompt", "Describe the color.")
    init_prompt = cfg.get("init_prompt")
    device = args.device or cfg.get("device")
    save_tokens = bool(cfg.get("save_tokens", False))

    keep = load_keep_colors()
    names = args.datasets or cfg.get("datasets") or sorted(keep)
    missing = [n for n in names if n not in keep]
    if missing:
        raise SystemExit(
            f"Датасеты {missing} отсутствуют в {KEEP_COLORS.name}.\n"
            f"Доступны: {sorted(keep)}.\n"
            "Либо они исключены аудитом (bfd_p-c — ниже разрешения носителя), "
            "либо конфиг устарел: уберите их из поля \"datasets\"."
        )

    # ---- план работ: что осталось посчитать ----
    todo: Dict[str, List[int]] = {}
    for ds in names:
        idxs = keep[ds]
        if not args.overwrite:
            idxs = [i for i in idxs if not is_done(out_root / ds / str(i))]
        if args.limit:
            idxs = idxs[: args.limit]
        todo[ds] = idxs

    total = sum(len(v) for v in todo.values())
    print(f"модель:  {model_name}")
    print(f"промпт:  {prompt!r}")
    print(f"выход:   {out_root.relative_to(ROOT)}")
    print(f"device:  {device or 'auto'}\n")
    print(f"{'датасет':12s} {'в гамуте':>9s} {'готово':>7s} {'к расчёту':>10s}")
    print("-" * 42)
    for ds in names:
        done = len(keep[ds]) - len([i for i in keep[ds] if not is_done(out_root / ds / str(i))])
        print(f"{ds:12s} {len(keep[ds]):9d} {done:7d} {len(todo[ds]):10d}")
    print("-" * 42)
    print(f"{'ИТОГО':12s} {sum(len(keep[d]) for d in names):9d} {'':7s} {total:10d}\n")

    if args.dry_run:
        print("dry-run: модель не загружалась.")
        return 0
    if total == 0:
        print("Всё уже посчитано.")
        return 0

    import torch
    from utils.embeddings.embedding_extractor import EmbeddingsExtractor
    from utils.embeddings.images_loader import save_all, tensor_shape

    extractor = EmbeddingsExtractor(model_name=model_name, device=device,
                                    system_prompt=init_prompt)
    started = time.time()
    processed, failed = 0, []

    try:
        for ds in names:
            if not todo[ds]:
                continue
            dataset_dir = dataset_root / ds
            meta = color_meta(dataset_dir)
            (out_root / ds).mkdir(parents=True, exist_ok=True)

            for idx in tqdm(todo[ds], desc=ds, unit="цвет"):
                img_path = dataset_dir / f"{idx}.png"
                color_dir = out_root / ds / str(idx)
                try:
                    with Image.open(img_path) as im:
                        img = im.convert("RGB")
                    out = extractor.extract([img], prompt=prompt)
                    color_dir.mkdir(parents=True, exist_ok=True)
                    saved = save_all(color_dir, out, save_tokens)
                    m = meta.get(idx, {})
                    (color_dir / "manifest.json").write_text(json.dumps({
                        "dataset": ds,
                        "index": idx,
                        "image": img_path.relative_to(ROOT).as_posix(),
                        "model": model_name,
                        "prompt": prompt,
                        "init_prompt": init_prompt,
                        "answer": out.get("model_answer", ""),
                        "saved": saved,
                        "shapes": {k: tensor_shape(out.get(k)) for k in (
                            "vision_pooled_mean", "projected_pooled_mean",
                            "lm_pooled_mean", "visual_token_lens")},
                        "xyz": m.get("xyz"), "xyY": m.get("xyY"),
                        "srgb": m.get("srgb"), "rgb_255": m.get("rgb_255"),
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    processed += 1
                except Exception as exc:                       # noqa: BLE001
                    failed.append({"dataset": ds, "index": idx, "error": repr(exc)})
                    print(f"\n[ОШИБКА] {ds}/{idx}: {exc!r}")
    finally:
        extractor.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - started
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_run_log.json").write_text(json.dumps({
        "model": model_name, "prompt": prompt, "device": device,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "processed": processed, "failed": failed,
        "elapsed_sec": round(elapsed, 1),
        "sec_per_color": round(elapsed / max(processed, 1), 3),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nготово: {processed} цветов за {elapsed/60:.1f} мин "
          f"({elapsed/max(processed,1):.2f} с/цвет)")
    if failed:
        print(f"ОШИБОК: {len(failed)} — перезапустите ту же команду, "
              f"успешные цвета будут пропущены")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
