#!/usr/bin/env python3
"""
Esegue gli esperimenti di classificazione per la tesi.

Esperimenti:
    1. Train privato → Test privato (baseline)
    2. Train privato → Test internet raw
    3. Train privato → Test internet cropped
    4. Train combined → Test internet cropped
    5. Train combined → Test privato

Usage:
    python run_experiments.py --all
    python run_experiments.py --setup
    python run_experiments.py --preprocess
    python run_experiments.py --train
    python run_experiments.py --test
    python run_experiments.py --status
"""

import os
import sys
import shutil
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "experiments.log"
TIMING_FILE = LOG_DIR / "timing.csv"


def setup_global_logging():
    """Configure ALL Python logging to go to experiments.log"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File handler - writes to experiments.log
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # Root logger - captures ALL modules
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# Timing tracker
_timing_stack = {}

# Paths
DRIVE_ROOT = PROJECT_ROOT.parent / "drive" / "Immagini_Internet"
INTERNET_RAW_SOURCE = DRIVE_ROOT
INTERNET_CROPPED_SOURCE = DRIVE_ROOT / "CROPPED_by_ARUTA" / "immagini_croppate_online_padding"

DATASET_ROOT = PROJECT_ROOT / "dataset"
INTERNET_RAW_TARGET = DATASET_ROOT / "internet_raw"
INTERNET_CROPPED_TARGET = DATASET_ROOT / "internet_cropped"
COMBINED_TARGET = DATASET_ROOT / "combined"
RAW_TARGET = DATASET_ROOT / "raw"

ALL_CLASSES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. majellensis",
    "O. sphegodes",
    "O. sphegodes_Palena"
]

INTERNET_CLASSES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. sphegodes"
]

EXPERIMENTS = {
    1: {"name": "exp1_privato_privato", "train": "privato", "test": "privato", "test_classes": ALL_CLASSES},
    2: {"name": "exp2_privato_internet_raw", "train": "privato", "test": "internet_raw", "test_classes": INTERNET_CLASSES},
    3: {"name": "exp3_privato_internet_cropped", "train": "privato", "test": "internet_cropped", "test_classes": INTERNET_CLASSES},
    4: {"name": "exp4_combined_internet_cropped", "train": "combined", "test": "internet_cropped", "test_classes": INTERNET_CLASSES},
    5: {"name": "exp5_combined_privato", "train": "combined", "test": "privato", "test_classes": ALL_CLASSES}
}


def log(msg):
    """Log su console e file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{timestamp} | {msg}"
    print(line, flush=True)
    # Scrive anche su file con flush immediato
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def start_timer(phase_name):
    """Inizia a tracciare il tempo per una fase."""
    _timing_stack[phase_name] = time.time()
    log(f"[START] {phase_name}")


def end_timer(phase_name):
    """Ferma il tracciamento e salva il tempo."""
    if phase_name not in _timing_stack:
        return
    start_time = _timing_stack[phase_name]
    elapsed = time.time() - start_time
    del _timing_stack[phase_name]

    # Salva su CSV
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    header = not TIMING_FILE.exists()
    with open(TIMING_FILE, "a") as f:
        if header:
            f.write("phase,start_time,end_time,duration_seconds,duration_hms\n")
        start_dt = datetime.fromtimestamp(start_time)
        f.write(f"{phase_name},{start_dt.isoformat()},{datetime.now().isoformat()},{elapsed:.2f},{format_duration(elapsed)}\n")

    log(f"[END] {phase_name} - {format_duration(elapsed)}")


def format_duration(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def is_done(phase):
    return (PROJECT_ROOT / "experiments" / f".{phase}_done").exists()


def mark_done(phase):
    done_file = PROJECT_ROOT / "experiments" / f".{phase}_done"
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.touch()
    log(f"[OK] {phase}")


def clear_done(phase):
    done_file = PROJECT_ROOT / "experiments" / f".{phase}_done"
    if done_file.exists():
        done_file.unlink()


def count_images(path, classes):
    counts = {}
    for cls in classes:
        cls_dir = path / cls
        if cls_dir.exists():
            counts[cls] = len([f for f in cls_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
        else:
            counts[cls] = 0
    return counts


def show_status():
    print("\n" + "="*70)
    print("STATO ESPERIMENTI")
    print("="*70)

    phases = [
        ("setup_images", "Setup immagini internet"),
        ("setup_combined", "Setup combined dataset"),
        ("preprocess_train_privato", "Preprocessing train privato"),
        ("preprocess_train_combined", "Preprocessing train combined"),
        ("preprocess_test_privato", "Preprocessing test privato"),
        ("preprocess_test_internet_raw", "Preprocessing test internet raw"),
        ("preprocess_test_internet_cropped", "Preprocessing test internet cropped"),
        ("train_privato", "Training modello privato"),
        ("train_combined", "Training modello combined"),
    ]

    print("\nFASI:")
    for phase, desc in phases:
        status = "[x]" if is_done(phase) else "[ ]"
        print(f"  {status} {desc}")

    print("\nESPERIMENTI:")
    for num, exp in EXPERIMENTS.items():
        status = "[x]" if is_done(f"exp_{num}") else "[ ]"
        print(f"  {status} Exp {num}: {exp['name']}")

    print("\nIMMAGINI:")
    for name, path, classes in [
        ("Dataset privato", RAW_TARGET, ALL_CLASSES),
        ("Internet raw", INTERNET_RAW_TARGET, INTERNET_CLASSES),
        ("Internet cropped", INTERNET_CROPPED_TARGET, INTERNET_CLASSES),
        ("Combined", COMBINED_TARGET, ALL_CLASSES),
    ]:
        if path.exists():
            total = sum(count_images(path, classes).values())
            print(f"  {name}: {total} totali")

    print("\n" + "="*70 + "\n")


def setup_folders():
    log("="*60)
    log("SETUP")
    log("="*60)

    for d in [INTERNET_RAW_TARGET, INTERNET_CROPPED_TARGET, COMBINED_TARGET,
              PROJECT_ROOT / "experiments",
              PROJECT_ROOT / "models" / "train_privato" / "checkpoints",
              PROJECT_ROOT / "models" / "train_combined" / "checkpoints"]:
        d.mkdir(parents=True, exist_ok=True)

    for exp in EXPERIMENTS.values():
        (PROJECT_ROOT / "results" / exp["name"]).mkdir(parents=True, exist_ok=True)

    copy_internet_images()
    create_combined()
    log("Setup completed")


def copy_internet_images():
    if is_done("setup_images"):
        return

    log("Copying internet images...")

    # Raw
    for class_name in INTERNET_CLASSES:
        source_name = class_name.replace(" ", "") + "_Int"
        source = INTERNET_RAW_SOURCE / source_name
        target = INTERNET_RAW_TARGET / class_name

        if source.exists():
            target.mkdir(parents=True, exist_ok=True)
            for img in source.iterdir():
                if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    shutil.copy2(img, target / img.name)

    # Cropped
    for class_name in INTERNET_CLASSES:
        source = INTERNET_CROPPED_SOURCE / class_name.replace(" ", "")
        target = INTERNET_CROPPED_TARGET / class_name

        if source.exists():
            target.mkdir(parents=True, exist_ok=True)
            for img in source.iterdir():
                if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    shutil.copy2(img, target / img.name)

    mark_done("setup_images")


def create_combined():
    if is_done("setup_combined"):
        return

    log("Creating combined dataset...")

    # Copy raw images
    for class_name in ALL_CLASSES:
        source = RAW_TARGET / class_name
        target = COMBINED_TARGET / class_name
        target.mkdir(parents=True, exist_ok=True)

        if source.exists():
            for img in source.iterdir():
                if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    shutil.copy2(img, target / img.name)

    # Add internet cropped images
    for class_name in INTERNET_CLASSES:
        source = INTERNET_CROPPED_TARGET / class_name
        target = COMBINED_TARGET / class_name

        if source.exists():
            for img in source.iterdir():
                if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    shutil.copy2(img, target / f"int_{img.name}")

    for class_name in ALL_CLASSES:
        target = COMBINED_TARGET / class_name
        count = len([f for f in target.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]) if target.exists() else 0
        log(f"  {class_name}: {count}")

    mark_done("setup_combined")


# === PREPROCESSING ===

def run_preprocessing():
    log("="*60)
    log("PREPROCESSING")
    log("="*60)

    from run_pipeline import run_augmentation, run_normalization, run_preprocessing as pipeline

    # Base config shared
    base_config = {
        'class_names': ALL_CLASSES,
        'train_size': 0.7,
        'validation_size': 0.1,
        'width': 256,
        'height': 512,
        'batch_size_preprocess': 32,
        'num_workers_preprocess': 4,
        'should_normalize': True,
        'seed': 57,
        'epochs': 26,
        'batch_size_train': 8,
        'num_workers_train': 4,
        'learning_rate': 0.001,
        'pretrained': True,
        'use_class_weights': False,
        'force_cpu': False,
    }

    # Train private - skip if already done
    if is_done("preprocess_train_privato"):
        log("[OK] preprocess_train_privato gia completato, skip")
    else:
        start_timer("preprocess_train_privato")
        log("Preprocessing train private...")
        config = base_config.copy()
        config.update({
            'raw_data_path': str(RAW_TARGET) + "/",
            'augmented_data_path': str(DATASET_ROOT / "augmented_privato") + "/",
            'processed_train_path': str(DATASET_ROOT / "train_privato.pt"),
            'processed_validation_path': str(DATASET_ROOT / "val_privato.pt"),
            'processed_test_path': str(DATASET_ROOT / "test_privato.pt"),
            'model_path': str(PROJECT_ROOT / "models" / "train_privato" / "model.pt"),
            'results_path': str(PROJECT_ROOT / "results" / "exp1_privato_privato") + "/",
        })
        run_augmentation(config)
        mean, std = run_normalization(config)
        pipeline(config, mean, std)
        end_timer("preprocess_train_privato")
        mark_done("preprocess_train_privato")

    # Train combined - skip if already done
    if is_done("preprocess_train_combined"):
        log("[OK] preprocess_train_combined gia completato, skip")
    else:
        start_timer("preprocess_train_combined")
        log("Preprocessing train combined...")
        config = base_config.copy()
        config.update({
            'raw_data_path': str(COMBINED_TARGET) + "/",
            'augmented_data_path': str(DATASET_ROOT / "augmented_combined") + "/",
            'processed_train_path': str(DATASET_ROOT / "train_combined.pt"),
            'processed_validation_path': str(DATASET_ROOT / "val_combined.pt"),
            'processed_test_path': str(DATASET_ROOT / "test_combined.pt"),
            'model_path': str(PROJECT_ROOT / "models" / "train_combined" / "model.pt"),
            'results_path': str(PROJECT_ROOT / "results" / "exp4_combined_internet_cropped") + "/",
        })
        run_augmentation(config)
        mean, std = run_normalization(config)
        pipeline(config, mean, std)
        end_timer("preprocess_train_combined")
        mark_done("preprocess_train_combined")

    # Test sets
    log("Preprocessing test sets...")
    from app.preprocess_data import preprocessData

    for test_type in ["privato", "internet_raw", "internet_cropped"]:
        if is_done(f"preprocess_test_{test_type}"):
            log(f"[OK] preprocess_test_{test_type} gia completato, skip")
            continue

        start_timer(f"preprocess_test_{test_type}")
        # Use augmented test folder for privato (already split), raw folders for internet
        if test_type == "privato":
            source = DATASET_ROOT / "augmented_privato" / "test"
        elif test_type == "internet_raw":
            source = INTERNET_RAW_TARGET
        else:
            source = INTERNET_CROPPED_TARGET
        classes = ALL_CLASSES if test_type == "privato" else INTERNET_CLASSES

        work = PROJECT_ROOT / "work" / test_type
        work.mkdir(parents=True, exist_ok=True)

        for cls in classes:
            src = source / cls
            dst = work / cls
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                for img in src.iterdir():
                    if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                        shutil.copy2(img, dst / img.name)

        preprocessData(
            str(work) + "/",
            str(DATASET_ROOT / f"test_{test_type}.pt"),
            256, 512, 32, 8, True, 57,
            "0.538217306137085 0.5433580875396729 0.368904173374176",
            "0.20864470303058624 0.23085597157478333 0.2029736042022705"
        )

        shutil.rmtree(work)
        end_timer(f"preprocess_test_{test_type}")
        mark_done(f"preprocess_test_{test_type}")

    log("Preprocessing completed")


# === TRAINING ===

def get_last_checkpoint(path):
    if not path.exists():
        return 0
    ckpts = list(path.glob("checkpoint_epoch_*.pt"))
    return max([int(c.stem.split("_")[-1]) for c in ckpts], default=0)


def train(train_type):
    phase = f"train_{train_type}"
    if is_done(phase):
        return

    start_timer(phase)
    log(f"Training {train_type}...")

    from run_pipeline import run_training

    model_dir = PROJECT_ROOT / "models" / f"train_{train_type}"
    ckpt_dir = model_dir / "checkpoints"

    config = {
        'processed_train_path': str(DATASET_ROOT / f"train_{train_type}.pt"),
        'processed_validation_path': str(DATASET_ROOT / f"val_{train_type}.pt"),
        'model_path': str(model_dir / "model.pt"),
        'class_names': ALL_CLASSES,
        'epochs': 26,
        'batch_size_train': 8,          # Same as original (Di Pierro)
        'learning_rate': 0.001,          # Same as original (Di Pierro)
        'num_workers_train': 4,
        'pretrained': True,
        'seed': 57,
        'use_class_weights': False,      # Same as original (Di Pierro)
        'force_cpu': False,
    }

    start = get_last_checkpoint(ckpt_dir)
    if start > 0:
        log(f"Resuming from epoch {start}")

    run_training(config, checkpoint_dir=str(ckpt_dir), start_epoch=start)
    end_timer(phase)
    mark_done(phase)


def run_training_phase():
    log("="*60)
    log("TRAINING")
    log("="*60)

    train("privato")
    train("combined")

    log("Training completed")


# === TESTING ===

def run_experiment(num):
    exp = EXPERIMENTS[num]
    phase = f"exp_{num}"

    if is_done(phase):
        return

    start_timer(phase)
    log(f"Experiment {num}: {exp['name']}")

    import torch
    from torch.utils.data import DataLoader
    from app.test_model import deviceLoader, modelLoader, testModel, generateOutputImages
    from app.preprocessing_tools.dataset_tool import getDatasetFromFile

    model_path = str(PROJECT_ROOT / "models" / f"train_{exp['train']}" / "model.pt")
    test_path = str(DATASET_ROOT / f"test_{exp['test']}.pt")
    results_dir = PROJECT_ROOT / "results" / exp["name"]

    with open(results_dir / "config.json", "w") as f:
        json.dump({
            "experiment": num,
            "name": exp["name"],
            "train": exp["train"],
            "test": exp["test"],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    device = deviceLoader()
    model = modelLoader(model_path, len(ALL_CLASSES), device)

    dataset = getDatasetFromFile(test_path)
    loader = DataLoader(dataset, 32, shuffle=True, num_workers=8)

    testModel(model, loader, device, exp["test_classes"], len(exp["test_classes"]))
    # Use ALL_CLASSES for visualization since model outputs all 6 classes
    generateOutputImages(dataset, model, device, ALL_CLASSES, str(results_dir), 30, 15)
    end_timer(phase)
    mark_done(phase)


def run_testing_phase():
    log("="*60)
    log("TESTING")
    log("="*60)

    for num in EXPERIMENTS:
        run_experiment(num)

    log("Testing completed")


# === MAIN ===

def run_all():
    start_timer("total_pipeline")

    start_timer("setup")
    setup_folders()
    end_timer("setup")

    start_timer("preprocessing")
    run_preprocessing()
    end_timer("preprocessing")

    start_timer("training")
    run_training_phase()
    end_timer("training")

    start_timer("testing")
    run_testing_phase()
    end_timer("testing")

    end_timer("total_pipeline")
    log("="*60)
    log("ALL EXPERIMENTS COMPLETED")
    log("="*60)


def main():
    setup_global_logging()
    parser = argparse.ArgumentParser(description="Esegue i 4 esperimenti")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset", type=str)

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.reset:
        clear_done(args.reset)
        log(f"Reset: {args.reset}")
    elif args.all:
        run_all()
    else:
        if args.setup:
            setup_folders()
        if args.preprocess:
            run_preprocessing()
        if args.train:
            run_training_phase()
        if args.test:
            run_testing_phase()

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    main()
