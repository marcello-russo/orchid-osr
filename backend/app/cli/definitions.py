# app/cli/definitions.py
"""
Constants and experiment definitions for the CLI orchestrator.
"""
from pathlib import Path

from app.config import settings

# Project root (backend/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dataset paths
DATASET_PATHS = {
    "privato": {
        "raw": "dataset/raw/",
        "augmented": "dataset/augmented_privato/",
        "train": "dataset/train_privato.pt",
        "val": "dataset/val_privato.pt",
        "test": "dataset/test_privato.pt",
    },
    "internet_raw": {
        "raw": "dataset/internet_raw/",
        "test": "dataset/test_internet_raw.pt",
    },
    "internet_cropped": {
        "raw": "dataset/internet_cropped/",
        "test": "dataset/test_internet_cropped.pt",
    },
    "combined": {
        "raw": "dataset/combined/",
        "augmented": "dataset/augmented_combined/",
        "train": "dataset/train_combined.pt",
        "val": "dataset/val_combined.pt",
        "test": "dataset/test_combined.pt",
    }
}

# Class definitions
ALL_CLASSES = settings.class_names

INTERNET_CLASSES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. sphegodes"
]

# Experiment definitions
EXPERIMENTS = {
    1: {
        "name": "exp1_privato_privato",
        "train_dataset": "privato",
        "test_dataset": "privato",
        "num_classes": 6,
        "test_classes": ALL_CLASSES,
        "description": "Baseline: train and test on private dataset (6 classes)"
    },
    2: {
        "name": "exp2_privato_internet_raw",
        "train_dataset": "privato",
        "test_dataset": "internet_raw",
        "num_classes": 6,
        "test_classes": INTERNET_CLASSES,
        "description": "Domain shift: train on private, test on internet raw"
    },
    3: {
        "name": "exp3_privato_internet_cropped",
        "train_dataset": "privato",
        "test_dataset": "internet_cropped",
        "num_classes": 6,
        "test_classes": INTERNET_CLASSES,
        "description": "Domain shift: train on private, test on internet cropped"
    },
    4: {
        "name": "exp4_combined_internet_cropped",
        "train_dataset": "combined",
        "test_dataset": "internet_cropped",
        "num_classes": 6,
        "test_classes": INTERNET_CLASSES,
        "description": "Best on internet: train on combined, test on internet cropped"
    },
    5: {
        "name": "exp5_combined_privato",
        "train_dataset": "combined",
        "test_dataset": "privato",
        "num_classes": 6,
        "test_classes": ALL_CLASSES,
        "description": "Catastrophic forgetting test: train on combined, test on private"
    }
}

# Default configuration
DEFAULT_CONFIG = {
    "model_variant": "resnet18",
    "epochs": 26,
    "batch_size": 8,
    "learning_rate": 0.001,
    "seed": 57,
    "pretrained": True,
    "num_folds": 5,
}
