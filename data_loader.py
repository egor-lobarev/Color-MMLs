"""
Munsell Chain Data Loader
-------------------------
Загружает эмбеддинги и метаданные из заранее вычисленных файлов.

Пример использования CLI:
    python data_loader.py --embeddings_dir data/embeddings/qwen2.5_7B/chroma_change \
                          --manifest_path data/colors/chroma_change/manifest.json \
                          --chain_index 0
"""

import json
import numpy as np
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional


class MunsellChainDataLoader:
    def __init__(self, embeddings_dir: Union[str, Path], manifest_path: Union[str, Path]):
        """
        embeddings_dir: путь к директории с эмбеддингами (подпапки с индексами)
        manifest_path: путь к manifest.json, где хранится описание цепочек и цветов
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.manifest_path = Path(manifest_path)

        if not self.embeddings_dir.exists():
            raise FileNotFoundError(f"embeddings_dir not found: {self.embeddings_dir}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest_path not found: {self.manifest_path}")

        # Загружаем manifest с метаданными
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.dataset_manifest = json.load(f)

        # Загружаем все эмбеддинги
        self.embeddings_data = self._load_embeddings(self.embeddings_dir)

        # Быстрая мапа: index -> позиция в списках embeddings_data['indices']
        self.index_to_pos = {idx: pos for pos, idx in enumerate(self.embeddings_data["indices"])}

        # Создаём mapping index -> metadata (из верхнего manifest)
        self.index_to_meta = self._build_index_metadata_map()

        # Организуем данные по цепочкам (списки эмбеддингов и метаданных)
        self.chains_embeddings, self.chains_metadata = self._organize_by_chain()


    def _load_embeddings(self, embeddings_dir: Path) -> Dict[str, list]:
        embeddings_data = {
            "lm_pooled_mean": [],
            "vision_pooled_mean": [],
            "visual_token_lens": [],
            "indices": [],
            "metadata": [],
        }

        # подпапки с цифровыми именами, сортируем по числу
        subdirs = sorted(
            [d for d in embeddings_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )

        for subdir in subdirs:
            idx = subdir.name
            manifest_path = subdir / "manifest.json"

            # metadata (локальный manifest в папке sample/)
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                embeddings_data["metadata"].append(manifest)
            else:
                embeddings_data["metadata"].append({})

            # Load embeddings, если нет — None
            def load_or_none(p: Path) -> Optional[np.ndarray]:
                if p.exists():
                    arr = np.load(p)
                    return arr.flatten()
                return None

            embeddings_data["lm_pooled_mean"].append(load_or_none(subdir / "lm_pooled_mean.npy"))
            embeddings_data["vision_pooled_mean"].append(load_or_none(subdir / "vision_pooled_mean.npy"))
            embeddings_data["visual_token_lens"].append(load_or_none(subdir / "visual_token_lens.npy"))
            embeddings_data["indices"].append(idx)

        print(f"✅ Loaded {len(embeddings_data['indices'])} embedding folders from {embeddings_dir}")
        return embeddings_data

    def _build_index_metadata_map(self) -> Dict[str, Dict[str, Any]]:
        index_to_color: Dict[str, Dict[str, Any]] = {}
        chains = self.dataset_manifest.get("chains", [])
        for chain in chains:
            for item in chain.get("items", []):
                idx = str(item.get("index"))
                meta = {
                    "notation": item.get("notation"),
                    "xyY": item.get("xyY"),
                    "H": item.get("H"),
                    "C": item.get("C"),
                    "V": item.get("V"),
                    "chain_description": chain.get("description"),
                }
                # RGB может быть в item
                if "RGB" in item:
                    meta["RGB"] = item.get("RGB")
                index_to_color[idx] = meta
        print(f"✅ Parsed metadata for {len(index_to_color)} color samples from manifest")
        return index_to_color

    # -------------------------------------------------
    # Группировка эмбеддингов по цепочкам
    # -------------------------------------------------
    def _organize_by_chain(self) -> Tuple[List[List[torch.Tensor]], List[List[Dict[str, Any]]]]:
        chains_embeddings: List[List[torch.Tensor]] = []
        chains_metadata: List[List[Dict[str, Any]]] = []

        chains = self.dataset_manifest.get("chains", [])
        for chain in chains:
            items = chain.get("items", [])
            chain_embs: List[torch.Tensor] = []
            chain_meta: List[Dict[str, Any]] = []

            for item in items:
                idx = str(item.get("index"))
                pos = self.index_to_pos.get(idx)
                if pos is None:
                    # нет папки с таким индексом -> пропускаем
                    # можно логировать, если нужно
                    # print(f"warning: index {idx} not found in embeddings_dir")
                    continue

                # берем vision, иначе lm
                vision_emb = self.embeddings_data["vision_pooled_mean"][pos]
                lm_emb = self.embeddings_data["lm_pooled_mean"][pos]

                chosen_emb = vision_emb if vision_emb is not None else lm_emb
                if chosen_emb is None:
                    # нет ни vision, ни lm
                    continue

                # преобразуем в torch tensor
                emb_tensor = torch.tensor(chosen_emb, dtype=torch.float32)
                chain_embs.append(emb_tensor)

                # берём метаданные (если нет — пустой dict)
                meta = self.index_to_meta.get(idx, {})
                chain_meta.append(meta)

            chains_embeddings.append(chain_embs)
            chains_metadata.append(chain_meta)

        print(f"✅ Organized {len(chains_embeddings)} chains (some chains may be empty if no embeddings found)")
        return chains_embeddings, chains_metadata


    def get_chain(self, index: int) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Возвращает эмбеддинги и метаданные по индексу цепочки"""
        if index < 0 or index >= len(self.chains_embeddings):
            raise IndexError(f"chain index {index} out of range [0, {len(self.chains_embeddings)-1}]")

        chain_embs = self.chains_embeddings[index]
        chain_meta = self.chains_metadata[index]

        if not chain_embs:
            raise ValueError(f"Chain #{index} has no embeddings (maybe missing files).")

        # torch.stack — все тензоры одной длины D
        emb_tensor = torch.stack(chain_embs, dim=0)  # shape (N, D)
        return emb_tensor, chain_meta


    def get_chains_data(self, chains_config):
        """
        Извлекает эмбеддинги и метаданные на основе конфигураций цепочек.

        chains_config: список словарей вида:
            {'fixed': {'H': 5.0, 'C': 8.0}, 'varying': [1.0, 2.0, 3.0]}

        Возвращает:
            results: список словарей:
                {
                    'embeddings': torch.Tensor [N, D],
                    'metadata': [ {H, C, V, xyY, RGB}, ... ]
                }
        """
        results = []

        for cfg in chains_config:
            fixed = cfg.get('fixed', {})
            varying = cfg.get('varying', None)

            # Определяем, какая компонента варьируется (из {H, C, V})
            varying_key = None
            if varying is not None:
                for key in ['H', 'C', 'V']:
                    if key not in fixed:
                        varying_key = key
                        break

            matched_embs = []
            matched_meta = []

            for chain_embs, chain_meta in zip(self.chains_embeddings, self.chains_metadata):
                if not chain_meta:
                    continue

                # Проверка, подходит ли цепочка под fixed
                def meta_matches_fixed(meta: Dict[str, Any]) -> bool:
                    for k, v in fixed.items():
                        val = meta.get(k)
                        if val is None:
                            return False
                        # если H — строка, сравниваем через str
                        if k == "H":
                            if str(val) != str(v):
                                return False
                        else:
                            try:
                                if not np.isclose(float(val), float(v), atol=1e-6):
                                    return False
                            except (TypeError, ValueError):
                                return False
                    return True

                # если нет хотя бы одного совпадения по fixed — пропускаем цепочку
                if not any(meta_matches_fixed(m) for m in chain_meta):
                    continue

                # Собираем элементы по varying
                selected_embs = []
                selected_meta = []
                for emb, meta in zip(chain_embs, chain_meta):
                    if not meta_matches_fixed(meta):
                        continue

                    if varying is not None and varying_key is not None:
                        val = meta.get(varying_key)
                        if val is None:
                            continue

                        # если varying_key это H (строка), сравниваем через str
                        if varying_key == "H":
                            if str(val) not in [str(x) for x in varying]:
                                continue
                        else:
                            try:
                                if not any(np.isclose(float(val), float(x), atol=1e-6) for x in varying):
                                    continue
                            except (TypeError, ValueError):
                                continue

                    # добавляем эмбеддинг
                    selected_embs.append(emb)

                    # добавляем метаданные, включая xyY и RGB если есть
                    meta_entry = {
                        "H": meta.get("H"),
                        "C": meta.get("C"),
                        "V": meta.get("V")
                    }
                    if "xyY" in meta:
                        meta_entry["xyY"] = meta["xyY"]
                    if "RGB" in meta:
                        meta_entry["RGB"] = meta["RGB"]
                    selected_meta.append(meta_entry)

                if selected_embs:
                    matched_embs.extend(selected_embs)
                    matched_meta.extend(selected_meta)

            if matched_embs:
                emb_tensor = torch.stack(matched_embs, dim=0)
                results.append({
                    'embeddings': emb_tensor,
                    'metadata': matched_meta
                })
            else:
                print(f"⚠️ Не найдено данных для {cfg}")

        print(f"✅ Извлечено {len(results)} цепочек из {len(chains_config)} конфигураций")
        return results
 

    def __len__(self) -> int:
        return len(self.chains_embeddings)


# ===================================================
# CLI
# ===================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Munsell Chain Data Loader CLI")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Путь к папке с эмбеддингами")
    parser.add_argument("--manifest_path", type=str, required=True, help="Путь к manifest.json")
    parser.add_argument("--chain_index", type=int, default=0, help="Индекс цепочки для вывода примера")
    args = parser.parse_args()

    loader = MunsellChainDataLoader(args.embeddings_dir, args.manifest_path)

    print(f"\nВсего цепочек: {len(loader)}")
    
    embs, meta = loader.get_chain(args.chain_index)


    print(f"\n✅ Цепочка #{args.chain_index}: {len(meta)} элементов, форма эмбеддингов {embs.shape}")
    print("Пример метаданных:")
    for m in meta[:3]:
        print(json.dumps(m, indent=2, ensure_ascii=False))

    chains_config = [
        {'fixed': {'H': 5.0, 'V': 3.0}, 'varying': [2.0, 4.0, 6.0, 8.0]},
        {'fixed': {'C': 6.0, 'V': 4.0}, 'varying': None}
    ]

    results = loader.get_chains_data(chains_config)

    for r in results:
        print("\nConfig:", r["config"])
        print("Embeddings shape:", r["embeddings"].shape)
        print("Metadata sample:", r["metadata"][:3])

