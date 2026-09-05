"""Shim: use `orchid ncd-comparison` (logic lives in app.core.ncd.comparison)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.ncd.comparison import main

if __name__ == "__main__":
    main()
