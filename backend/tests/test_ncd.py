import os
import tempfile

import numpy as np
import pytest

from app.api.routes.ncd import _load_npy_or_none


def test_load_npy_success():
    """Loading a valid .npy file returns its contents as a list."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, arr)
        path = f.name
    try:
        result = _load_npy_or_none(path)
        assert result == [[1.0, 2.0], [3.0, 4.0]]
    finally:
        os.unlink(path)


def test_load_npy_not_found():
    """Missing file returns None (not raises)."""
    result = _load_npy_or_none("/nonexistent/path.npy")
    assert result is None


def test_load_npy_invalid():
    """Corrupted .npy file returns None after logging warning."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        f.write(b"not a valid npy file")
        path = f.name
    try:
        result = _load_npy_or_none(path)
        assert result is None
    finally:
        os.unlink(path)
