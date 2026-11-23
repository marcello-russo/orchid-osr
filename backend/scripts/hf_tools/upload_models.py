"""Shim: use `orchid hf upload-models` (logic lives in app.hf.models)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.hf.models import main

if __name__ == "__main__":
    main()
