"""Shim: use `orchid hf download` (logic lives in app.hf.sync)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.hf.sync import download_dataset


def main():
    download_dataset(sample_ratio=float(os.environ.get("DATASET_SAMPLE_RATIO", "1.0")))


if __name__ == "__main__":
    main()
