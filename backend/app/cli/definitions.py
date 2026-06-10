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
    },
    "clean": {
        "train": "dataset/cropped_256x512/train/",
        "test": "dataset/cropped_256x512/test/",
    },
}

# Class definitions — single source of truth is app.config
ALL_CLASSES = settings.class_names
INTERNET_CLASSES = settings.internet_classes

# Classes used for training (all 6 known, O. sphegodes_Napoli is only in test as novel class)
TRAIN_CLASSES = [c for c in ALL_CLASSES if c != "O. sphegodes_Napoli"]

# Experiment definitions
# NOTE: Exp 1-5 have data leakage issues (confirmed for exp4 and exp5, likely for exp1-3).
# Exp 4: combined contains internet_cropped images → test on internet_cropped = test on seen data.
# Exp 5: combined contains all private images → test on privato = test on seen data.
# Exp 6 uses a clean split with no leakage (50 test/class held out from training).
EXPERIMENTS = {
    # Experiments 1-5 removed — invalidated by data leakage (MD5 duplicates between train/test).
    # See git history or AUDIT.md for details.
    6: {
        "name": "exp6_clean_split",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "description": "Clean split: no data leakage, balanced test (50/class), all 6 classes",
    },
    7: {
        "name": "exp7_supcon",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "supcon",
        "description": "Supervised Contrastive pretraining + CE fine-tuning (Exp 7)",
    },
    8: {
        "name": "exp8_sphor",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "supcon",
        "description": "SupCon + Spherical Orthogonal Prototypes (SpHOR-style, Exp 8)",
    },
     9: {
        "name": "exp9_ncd",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "supcon",
        "ncd_mode": True,
        "description": "SpHOR with O. sphegodes_Napoli as hidden novel class (NCD)",
    },
    10: {
        "name": "exp10_mgh_ncd",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "supcon",
        "ncd_mode": True,
        "description": (
            "MGH-NCD: Multi-Granularity Hard-negative Novel Class Discovery. "
            "6 known species; O. sphegodes_Napoli is the novel class (test only)."
        ),
    },
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
