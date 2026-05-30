# app/preprocessing/augment.py
"""Augmentation functions for k-fold training."""

from app.preprocessing import rotation


def get_augmentation_functions():
    """Return standard augmentation functions for fold training."""
    return [
        rotation.identity,
        rotation.horizontal_flip,
        rotation.vertical_flip,
        rotation.horizontal_and_vertical_flip,
    ]
