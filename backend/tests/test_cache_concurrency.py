"""Test per la thread-safety della cache modelli in deps.py."""

import threading
from unittest.mock import patch, MagicMock

import pytest


class TestModelCacheConcurrency:
    """Verifica che _cache_lock prevenga race condition."""

    @patch("app.api.deps.create_model")
    @patch("app.api.deps._find_weights")
    @patch("app.api.deps.available_models")
    def test_concurrent_access_does_not_crash(self, mock_available, mock_find, mock_create):
        """Due thread che caricano lo stesso modello non devono crashare."""
        mock_available.return_value = ["resnet18", "resnet50"]
        from app.api.deps import get_model, _model_cache, MAX_CACHED_MODELS

        _model_cache.clear()

        mock_find.return_value = None
        mock_create.return_value = MagicMock()

        errors = []

        def load_model():
            try:
                get_model("resnet18")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_model) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access caused errors: {errors}"
        # Model should be cached once
        assert len(_model_cache) == 1

    @patch("app.api.deps.create_model")
    @patch("app.api.deps._find_weights")
    @patch("app.api.deps.available_models")
    def test_cache_eviction_with_lock(self, mock_available, mock_find, mock_create):
        """Cache che supera MAX_CACHED_MODELS deve fare eviction senza crash."""
        mock_available.return_value = ["a", "b", "c", "d", "e", "f", "g"]
        from app.api.deps import get_model, _model_cache, MAX_CACHED_MODELS

        _model_cache.clear()
        mock_find.return_value = None

        models = [MagicMock() for _ in range(MAX_CACHED_MODELS + 2)]
        mock_create.side_effect = models

        errors = []

        def load_model(idx):
            try:
                get_model(chr(97 + idx))  # a, b, c, d, e, f, g
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_model, args=(i,)) for i in range(MAX_CACHED_MODELS + 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Eviction caused errors: {errors}"
        # Cache should not exceed MAX_CACHED_MODELS
        assert len(_model_cache) <= MAX_CACHED_MODELS

    @patch("app.api.deps.create_model")
    @patch("app.api.deps._find_weights")
    @patch("app.api.deps.available_models")
    def test_cache_hit_does_not_duplicate(self, mock_available, mock_find, mock_create):
        """Stesso modello richiesto più volte non deve creare duplicati."""
        mock_available.return_value = ["resnet18", "resnet50"]
        from app.api.deps import get_model, _model_cache

        _model_cache.clear()
        mock_find.return_value = None
        mock_create.return_value = MagicMock()

        # Prima richiesta: carica
        get_model("resnet18")
        assert mock_create.call_count == 1
        assert len(_model_cache) == 1

        # Seconda richiesta: cache hit
        get_model("resnet18")
        assert mock_create.call_count == 1  # non deve ricaricare
        assert len(_model_cache) == 1
