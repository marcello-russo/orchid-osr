"""Shim: use `orchid ncd --stats` (logic lives in app.core.ncd.stats)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.ncd.stats import main

if __name__ == "__main__":
    main()
