"""Stratified batch sampler: garantisce ≥2 campioni per classe per batch."""

import random

from torch.utils.data import Sampler


class StratifiedBatchSampler(Sampler[int]):
    """Garantisce almeno 2 campioni per classe in ogni batch SupCon.

    Per batch: seleziona classi con ≥2 campioni, prende 2 per classe,
    riempe il resto casualmente. Previene batch degeneri (zero gradienti).
    """

    def __init__(self, labels: list[int], batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_indices: dict[int, list[int]] = {}
        for idx, lbl in enumerate(labels):
            self.class_indices.setdefault(lbl, []).append(idx)

    def __iter__(self):
        pool = [list(c) for c in self.class_indices.values()]
        if self.shuffle:
            for ci in pool:
                random.shuffle(ci)
        batches = []
        while sum(len(c) for c in pool) >= self.batch_size:
            batch = []
            for ci in pool:
                if len(ci) >= 2 and len(batch) + 2 <= self.batch_size:
                    batch.extend(ci[:2])
                    del ci[:2]
            remaining = [i for ci in pool for i in ci]
            if self.shuffle:
                random.shuffle(remaining)
            needed = self.batch_size - len(batch)
            batch.extend(remaining[:needed])
            used = set(remaining[:needed])
            for ci in pool:
                ci[:] = [i for i in ci if i not in used]
            batches.append(batch)
            pool = [ci for ci in pool if ci]
        return iter(batches)

    def __len__(self):
        total = sum(len(c) for c in self.class_indices.values())
        return max(0, total // self.batch_size)
