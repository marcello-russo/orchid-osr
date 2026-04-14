"""Load experiment data from JSON files."""
import json
import os
from glob import glob


def load_experiment_data(exp_dir):
    """
    Load all data for a k-fold experiment from its directory.

    Args:
        exp_dir: path to directory containing config.json, fold_*.json, results.json

    Returns:
        dict with keys: config, folds (list of fold dicts), results

    Raises:
        FileNotFoundError: if exp_dir or required files don't exist
    """
    exp_dir = str(exp_dir)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Directory not found: {exp_dir}")

    # Load config
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {exp_dir}")
    with open(config_path) as f:
        config = json.load(f)

    # Load fold files
    fold_files = sorted(glob(os.path.join(exp_dir, "fold_*.json")))
    if not fold_files:
        raise FileNotFoundError(f"No fold_*.json files found in {exp_dir}")
    folds = []
    for ff in fold_files:
        with open(ff) as f:
            folds.append(json.load(f))

    # Load results (optional)
    results_path = os.path.join(exp_dir, "results.json")
    results = None
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    return {
        "config": config,
        "folds": folds,
        "results": results,
    }
