from pathlib import Path
from typing import Any

from app.config import settings

# Project root (backend/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

ALL_CLASSES = settings.class_names
INTERNET_CLASSES = settings.internet_classes
# Le sei classi note. La novel class di Exp 4 (O. garganica) viene esclusa
# dal training tramite exclude_classes, non da questa lista.
TRAIN_CLASSES = list(ALL_CLASSES)

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

# I valori sono eterogenei per esperimento (int, str, liste, dict): il tipo
# preciso sarebbe un TypedDict con una variante per exp, cioè più codice del
# problema. `Any` qui è onesto — il CLI valida i campi dove li usa.
DEFAULT_CONFIG: dict[str, Any] = {
    "model_variant": "resnet18",
    "epochs": 26,
    "batch_size": 8,
    "learning_rate": 0.001,
    "seed": 57,  # aligned with MASTER_SEED in config.py
    "pretrained": True,
    "num_folds": 5,
}


EXPERIMENTS: dict[int, dict[str, Any]] = {
    1: {
        "name": "exp1_ce",
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
    2: {
        "name": "exp2_supcon",
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
        "description": "Supervised Contrastive pretraining + CE fine-tuning (Exp 2)",
    },
    3: {
        "name": "exp3_sphor",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 6,
        "test_classes": TRAIN_CLASSES,
        "loss_type": "supcon",
        "ncd_mode": False,
        "phases": ["SupCon + SpHOR", "CE fine-tune"],
        "recipe": "Due fasi: (1) SupCon + SpHOR: come Exp 2 ma con prototipi ortogonali sferici "
        "(Spherical Orthogonal Prototypes). Loss addizionale repulsiva "
        "L_rep = (1/K(K-1)) Σ_{i≠j} cos(θ_ij)² che forza i prototipi "
        "ad essere ortogonali, massimizzando la copertura angolare sulla sfera. "
        "(2) CE fine-tuning: backbone congelato, testa lineare. "
        "Closed-set: tutte le 6 classi note, nessuna novel class.",
        "description": "SupCon + Spherical Orthogonal Prototypes (SpHOR-style, Exp 3, closed-set)",
    },
    4: {
        "name": "exp4_sphor_ncd",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 5,
        "test_classes": TRAIN_CLASSES,
        "exclude_classes": ["O. garganica"],
        "loss_type": "supcon",
        "ncd_mode": True,
        "phases": ["SupCon + SpHOR", "CE fine-tune"],
        "recipe": "Identico a Exp 3 (150 epoche, patience 15 in entrambe le "
        "fasi). Novità: O. garganica è esclusa dal training (5 classi note) "
        "e usata solo in test come novel class per la valutazione OSR.",
        "description": "SpHOR with O. garganica as hidden novel class (NCD)",
    },
    5: {
        "name": "exp5_sphor_ncd",
        "train_dataset": "clean",
        "test_dataset": "clean",
        "num_classes": 4,
        "test_classes": [c for c in TRAIN_CLASSES if c not in ["O. garganica", "O. majellensis"]],
        "exclude_classes": ["O. garganica", "O. majellensis"],
        "loss_type": "supcon",
        "ncd_mode": True,
        "phases": ["SupCon + SpHOR", "CE fine-tune"],
        "num_epochs": 150,
        "ce_epochs": 100,
        "recipe": "Identico a Exp 4 ma con 2 novel nascoste (garganica+majellensis).",
        "description": "SpHOR with 2 hidden novel classes (garganica+majellensis)",
    },
}

# MGH escluso dalla narrazione 1-5
EXPERIMENTS_ARCHIVED = {
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

ALL_SPECIES = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]


def parse_novel_arg(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    normed = []
    for p in parts:
        low = p.lower().replace("o.", "").strip()
        # match esatto dopo normalizzazione
        match = next((sp for sp in ALL_SPECIES if low == sp.lower().replace("o.", "").strip()), None)
        if not match:
            raise ValueError(f"Specie '{p}' non trovata. Valide: {ALL_SPECIES}")
        normed.append(match)
    return normed


def resolve_experiment(exp_id: int, novel_override: list[str] | None = None) -> dict[str, Any]:
    import copy

    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Exp {exp_id} non trovato")
    cfg = copy.deepcopy(EXPERIMENTS[exp_id])
    if novel_override is not None:
        if len(novel_override) >= 6:
            raise ValueError("Troppe novel class")
        for cls in novel_override:
            if cls not in ALL_SPECIES:
                raise ValueError(f"Specie '{cls}' non trovata. Valide: {ALL_SPECIES}")
        cfg["exclude_classes"] = novel_override
        cfg["num_classes"] = 6 - len(novel_override)
        cfg["test_classes"] = [c for c in TRAIN_CLASSES if c not in novel_override]
    return cfg
