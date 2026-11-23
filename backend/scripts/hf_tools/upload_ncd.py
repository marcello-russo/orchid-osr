"""Shim: use `orchid hf upload-ncd` (logic lives in app.hf.ncd)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.hf.ncd import main

if __name__ == "__main__":
    main()
