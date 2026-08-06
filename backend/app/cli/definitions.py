from pathlib import Path

from app.config import settings

# Project root (backend/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

ALL_CLASSES = settings.class_names
INTERNET_CLASSES = settings.internet_classes
TRAIN_CLASSES = [c for c in ALL_CLASSES if c != "O. sphegodes_Napoli"]

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

DEFAULT_CONFIG = {
    "model_variant": "resnet18",
    "epochs": 26,
    "batch_size": 8,
    "learning_rate": 0.001,
    "seed": 57,  # aligned with MASTER_SEED in config.py
    "pretrained": True,
    "num_folds": 5,
}


EXPERIMENTS = {
    6: {
        "name": "exp6_clean_split",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "ce",
        "phases": ["CE"],
        "recipe": "Standard Cross-Entropy training con backbone intero. "
        "Una fase sola: loss CE su 6 classi. Backbone e testa addestrati insieme. "
        "150 epoche, Adam, LR 0.001, batch 8. Baseline per confronto.",
        "description": "Clean split: no data leakage, balanced test (50/class), all 6 classes",
    },
    7: {
        "name": "exp7_supcon",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "supcon",
        "phases": ["SupCon", "CE fine-tune"],
        "recipe": "Due fasi: (1) SupCon contrastive pretraining: organizza lo spazio di embedding "
        "attraendo campioni della stessa specie e respingendo specie diverse. "
        "Loss: SupConLoss con temperatura τ=0.07, proiezione a 128 dim. "
        "Backbone + projection head addestrati. "
        "(2) CE fine-tuning: backbone congelato, sola testa lineare addestrata "
        "con CrossEntropyLoss. AdamW, LR 0.001 per SupCon, 0.01 per CE.",
        "description": "Supervised Contrastive pretraining + CE fine-tuning (Exp 7)",
    },
    8: {
        "name": "exp8_sphor",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "supcon",
        "ncd_mode": False,
        "phases": ["SupCon + SpHOR", "CE fine-tune"],
        "recipe": "Due fasi: (1) SupCon + SpHOR: come Exp 7 ma con prototipi ortogonali sferici "
        "(Spherical Orthogonal Prototypes). Loss addizionale repulsiva "
        "L_rep = (1/K(K-1)) Σ_{i≠j} cos(θ_ij)² che forza i prototipi "
        "ad essere ortogonali, massimizzando la copertura angolare sulla sfera. "
        "(2) CE fine-tuning: backbone congelato, testa lineare. "
        "Closed-set: tutte le 6 classi note, nessuna novel class.",
        "description": "SupCon + Spherical Orthogonal Prototypes (SpHOR-style, Exp 8, closed-set)",
    },
    9: {
        "name": "exp9_ncd",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 5,
        "test_classes": TRAIN_CLASSES,
        "exclude_classes": ["O. garganica"],
        "loss_type": "supcon",
        "ncd_mode": False,
        "phases": ["SupCon + SpHOR", "CE fine-tune"],
        "recipe": "Identico a Exp 8 (150 epoche, patience 15 in entrambe le "
        "fasi). Novità: O. garganica è esclusa dal training (5 classi note) "
        "e usata solo in test come novel class per la valutazione OSR.",
        "description": "SpHOR with O. garganica as hidden novel class (NCD)",
    },
    10: {
        "name": "exp10_mgh_ncd",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "supcon",
        "ncd_mode": True,
        "phases": ["MGH-SupCon", "CE fine-tune"],
        "recipe": "Due fasi con Multi-Granularity Hard-negative mining: "
        "(1) SupCon modificato: coppie (majellensis, sphegodes) pesate 5×, "
        "(sphegodes, sphegodes_Palena) pesate 3× — forza la separazione "
        "delle specie più simili. "
        "Features multi-granularità: per ViT (DINOv2) si concatenano "
        "CLS finale + patch pool + CLS intermedio; per CNN (ResNet, ConvNeXt) "
        "si concatenano features penultimo e ultimo stadio. "
        "Backbone congelato dopo SupCon. "
        "(2) CE fine-tuning: testa lineare, backbone frozen.",
        "description": (
            "MGH-NCD: Multi-Granularity Hard-negative Novel Class Discovery. "
            "6 known species; O. sphegodes_Napoli is the novel class (test only)."
        ),
    },
}
