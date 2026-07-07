"""Tests for StratifiedBatchSampler."""

from collections import Counter

from app.core.kfold.sampler import StratifiedBatchSampler


class TestStratifiedBatchSampler:
    def test_every_batch_has_at_least_two_of_some_class(self):
        labels = [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5]
        sampler = StratifiedBatchSampler(labels, batch_size=8)
        for batch_indices in sampler:
            batch_labels = [labels[i] for i in batch_indices]
            assert len(batch_indices) == 8
            counts = Counter(batch_labels)
            assert any(c >= 2 for c in counts.values()), f"Batch degenerato: {counts}"

    def test_all_indices_are_within_range(self):
        labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        sampler = StratifiedBatchSampler(labels, batch_size=4)
        for batch_indices in sampler:
            for idx in batch_indices:
                assert 0 <= idx < len(labels)

    def test_len_returns_reasonable_batch_count(self):
        labels = list(range(12))
        sampler = StratifiedBatchSampler(labels, batch_size=4)
        assert len(sampler) == 3
