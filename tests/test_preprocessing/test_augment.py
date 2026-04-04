# tests/test_preprocessing/test_augment.py
from app.preprocessing.augment import get_augmentation_functions


def test_get_augmentation_functions_returns_list():
    funcs = get_augmentation_functions()
    assert isinstance(funcs, list)
    assert len(funcs) == 4


def test_get_augmentation_functions_are_callable():
    funcs = get_augmentation_functions()
    for f in funcs:
        assert callable(f)
