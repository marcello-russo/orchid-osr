"""Shim: use `orchid hf sync` (logic lives in app.hf.sync)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.hf.sync import main

if __name__ == "__main__":
    main()
