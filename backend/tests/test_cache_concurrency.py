"""Test per la thread-safety della cache modelli in deps.py."""

import threading
from unittest.mock import MagicMock, patch


class TestModelCacheConcurrency:
    """Verifica che _cache_lock prevenga race condition."""

    @patch("app.api.cache._find_weights")
    @patch("app.core.models.available_models")
    @patch("app.core.models.create_model")
    def test_concurrent_access_does_not_crash(self, mock_create, mock_available, mock_find):
        """Due thread che caricano lo stesso modello non devono crashare."""
        mock_available.return_value = ["resnet18", "resnet50"]
        from app.api.cache import _model_cache, get_model

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
        assert len(_model_cache) == 1

    @patch("app.api.cache._find_weights")
    @patch("app.core.models.available_models")
    @patch("app.core.models.create_model")
    def test_cache_eviction_with_lock(self, mock_create, mock_available, mock_find):
        """Cache che supera MAX_CACHED_MODELS deve fare eviction senza crash."""
        mock_available.return_value = ["a", "b", "c", "d", "e", "f", "g"]
        from app.api.cache import MAX_CACHED_MODELS, _model_cache, get_model

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
        assert len(_model_cache) <= MAX_CACHED_MODELS

    @patch("app.api.cache._find_weights")
    @patch("app.core.models.available_models")
    @patch("app.core.models.create_model")
    def test_cache_hit_does_not_duplicate(self, mock_create, mock_available, mock_find):
        """Stesso modello richiesto più volte non deve creare duplicati."""
        mock_available.return_value = ["resnet18", "resnet50"]
        from app.api.cache import _model_cache, get_model

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
