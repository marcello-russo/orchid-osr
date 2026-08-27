#!/usr/bin/env python3
"""Rinomina esperimenti 6-9 -> 1-4.

Uso:
    python scripts/rename_experiments.py            # dry-run
    python scripts/rename_experiments.py --apply    # esegue
"""

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"

OLD_TO_NEW = {
    "exp1_ce": "exp1_ce",
    "exp2_supcon": "exp2_supcon",
    "exp3_sphor": "exp3_sphor",
    "exp4_sphor_ncd": "exp4_sphor_ncd",
}

DATASET_RENAMES = {
    "train_ncd_exp4": "train_ncd_exp4",
    "test_ncd_exp4": "test_ncd_exp4",
}

EXTENSIONS = {".py", ".ts", ".tsx", ".json", ".sh", ".md", ".tex", ".yml", ".yaml", ".txt"}

REPLACEMENTS = [
    ("exp1_ce", "exp1_ce"),
    ("exp2_supcon", "exp2_supcon"),
    ("exp3_sphor", "exp3_sphor"),
    ("exp4_sphor_ncd", "exp4_sphor_ncd"),
    ("train_ncd_exp4", "train_ncd_exp4"),
    ("test_ncd_exp4", "test_ncd_exp4"),
    ("Exp 1 (CE)", "Exp 1 (CE)"),
    ("Exp 2 (SupCon)", "Exp 2 (SupCon)"),
    ("Exp 3 (SpHOR)", "Exp 3 (SpHOR)"),
    ("Exp 4 (SpHOR-NCD)", "Exp 4 (SpHOR-NCD)"),
    ("Exp 6 \\xe2\\x80\\x94", "Exp 1 \\xe2\\x80\\x94"),
    ("Exp 7 \\xe2\\x80\\x94", "Exp 2 \\xe2\\x80\\x94"),
    ("Exp 8 \\xe2\\x80\\x94", "Exp 3 \\xe2\\x80\\x94"),
    ("Exp 9 \\xe2\\x80\\x94", "Exp 4 \\xe2\\x80\\x94"),
    ("Exp 1-4", "Exp 1-4"),
    ("Exp 1--4", "Exp 1--4"),
    ('"exp1"', '"exp1"'),
    ("'exp1'", "'exp1'"),
    ('"exp2"', '"exp2"'),
    ("'exp2'", "'exp2'"),
    ('"exp3"', '"exp3"'),
    ("'exp3'", "'exp3'"),
    ('"exp4"', '"exp4"'),
    ("'exp4'", "'exp4'"),
    ("-1-", "-1-"),
    ("-2-", "-2-"),
    ("-3-", "-3-"),
    ("-4-", "-4-"),
    ("choices=[1, 2, 3, 4, 5, 10]", "choices=[1, 2, 3, 4, 5, 10]"),
]


def should_skip(path: Path) -> bool:
    rel = str(path.relative_to(REPO_ROOT))
    skip = [
        ".git", "node_modules", "__pycache__", ".pytest_cache",
        ".ruff_cache", "venv", ".env", "*.bak",
        "*.png", "*.jpg", "*.jpeg", "*.webp",
        "*.pth", "*.safetensors", "*.pt", "*.npy", "*.pdf",
        "*.aux", "*.bbl", "*.bcf", "*.blg", "*.fdb_latexmk",
        "*.fls", "*.log", "*.out", "*.run.xml", "*.toc",
        "*.lof", "*.lot", "*.pyc", "*.pyo",
    ]
    for p in skip:
        if p.startswith("*") and str(path).endswith(p[1:]):
            return True
        if p in rel:
            return True
    return False


def replace_in_file(filepath: Path, dry_run: bool = True) -> int:
    if should_skip(filepath):
        return 0
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0
    new_content = content
    n = 0
    for old, new in REPLACEMENTS:
        if old in new_content:
            new_content = new_content.replace(old, new)
            n += 1
    if n > 0 and not dry_run:
        filepath.write_text(new_content, encoding="utf-8")
        print(f"  {filepath.relative_to(REPO_ROOT)}: {n} replacements")
    return n


def process_dir(root: Path, dry_run: bool) -> int:
    total = 0
    for f in root.rglob("*"):
        if f.is_file() and f.suffix in EXTENSIONS:
            total += replace_in_file(f, dry_run)
    return total


def rename_dirs(base: Path) -> list:
    result = []
    if not base.exists():
        return result
    for old_pref, new_pref in OLD_TO_NEW.items():
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith(old_pref):
                new_name = d.name.replace(old_pref, new_pref, 1)
                result.append((d, d.parent / new_name))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dry_run = not args.apply

    print("=" * 60)
    print("RENAME EXPERIMENTS 6-9 -> 1-4")
    print(f"Mode: {'APPLY' if not dry_run else 'DRY-RUN'}")
    print("=" * 60)

    print("\n[1/3] Processing code files...")
    n = process_dir(REPO_ROOT, dry_run)
    if dry_run:
        print(f"  (replacements would be applied in {n} files)")

    print("\n[2/3] Renaming experiment directories...")
    for old, new in rename_dirs(BACKEND / "experiments"):
        if dry_run:
            print(f"  {old.name} -> {new.name}")
        else:
            os.rename(str(old), str(new))
            print(f"  {old.name} -> {new.name}")

    print("\n[3/3] Renaming dataset directories...")
    ds = BACKEND / "dataset"
    for old_name, new_name in DATASET_RENAMES.items():
        for size in ["", "cropped_224x224", "cropped_256x512"]:
            old_path = ds / size / old_name if size else ds / old_name
            if old_path.exists():
                new_path = ds / size / new_name if size else ds / new_name
                if dry_run:
                    print(f"  {old_path} -> {new_path}")
                else:
                    os.rename(str(old_path), str(new_path))
                    print(f"  {old_path} -> {new_path}")

    print("\n" + "=" * 60)
    print("DRY-RUN done, use --apply to apply" if dry_run else "RENAME done")
    print("=" * 60)


if __name__ == "__main__":
    main()
