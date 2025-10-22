"""
Munsell Chain Data Loader

Пример использования:
    python data_loader.py --config config_path --chain_index 0
"""
import json
import numpy as np
import torch
from typing import List, Dict, Union, Optional
import argparse
import sys
import argparse

def dummy_embed_func(H: float, C: float, V: float) -> torch.Tensor:
    """
    Простая детерминированная функция-заглушка эмбеддинга.
    (в будущем заменить на вызов Qwen с усреднением визуального/текстового слоя)
    """
    x = torch.tensor([H / 100.0, C / 10.0, V / 10.0], dtype=torch.float32)
    return torch.cat([torch.sin(x * 3.14), torch.cos(x * 3.14)])



class MunsellChainDataLoader:
    def __init__(
        self,
        chains_config,
        embed_func
    ):
        """
        chains_config: список конфигураций цепочек
        embed_func: функция, которая получает (H, C, V) и возвращает эмбеддинг
        """
        self.chains_config = chains_config
        self.embed_func = embed_func

        # Основные хранилища
        self.embeddings: List[List[torch.Tensor]] = []  # [[tensor,...], [tensor,...], ...]
        self.metadata: List[List[Dict[str, float]]] = []  # [[{H,C,V},...], ...]

        # При инициализации сразу создаём все данные
        self._prepare_all_chains()

    # Генерация цепочки метаданных
    def _generate_metadata(
        self, fixed: Dict[str, float], varying: Dict[str, List[float]]
    ) -> List[Dict[str, float]]:
        keys = ["H", "C", "V"]
        vary_key = list(varying.keys())[0]
        vary_values = varying[vary_key]
        metadata = []
        for val in vary_values:
            entry = {k: (val if k == vary_key else fixed.get(k, 0.0)) for k in keys}
            metadata.append(entry)
        return metadata

    # Подготовка всех цепочек
    def _prepare_all_chains(self):
        """Создаёт эмбеддинги и метаданные для всех цепочек"""
        for chain_cfg in self.chains_config:
            fixed = chain_cfg.get("fixed", {})
            varying = chain_cfg.get("varying", {})

            # генерируем метаданные
            meta = self._generate_metadata(fixed, varying)

            # генерируем эмбеддинги (через переданную функцию)
            chain_embs = [self.embed_func(m["H"], m["C"], m["V"]) for m in meta]

            # сохраняем
            self.embeddings.append(chain_embs)
            self.metadata.append(meta)

    # Получение цепочки по индексу
    def get_chain(self, index: int):
        """
        Возвращает эмбеддинги и метаданные по индексу цепочки.
        """
        chain_embs = torch.stack(self.embeddings[index])
        chain_meta = self.metadata[index]
        return chain_embs, chain_meta

    # Кол-во цепочек
    def __len__(self):
        return len(self.chains_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Munsell Chain Data Loader CLI")
    parser.add_argument("--config", type=str, required=True, help="Путь к JSON конфигурации цепочек")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # создаём DataLoader с dummy функцией
    loader = MunsellChainDataLoader(cfg, embed_func=dummy_embed_func)

    print(f"✅ Загружено {len(loader)} цепочек")
    for i in range(len(loader)):
        embs, meta = loader.get_chain(i)
        print(f"\n Цепочка #{i}: {len(meta)} цветов, эмбеддинг формы {embs.shape}")
        print("Пример метаданных:", meta[:3])
        print("Пример эмбеддингов:\n", embs[:2])