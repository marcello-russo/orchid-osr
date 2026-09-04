# ruff: noqa: D100,D103,D104,D105,D107,D401,PLR0912,PLR0915,PLR2004,E501
"""Sync figures to frontend."""

from __future__ import annotations

import shutil
from pathlib import Path


def sync_frontend(src: Path, dst: Path) -> None:
    """Copy png/webp files from *src* to *dst* via shutil.copy2."""
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.webp"):
        for p in src.glob(pattern):
            shutil.copy2(p, dst / p.name)


def build(*a, **kw):
    # keep stub compat
    if len(a) >= 2:
        return sync_frontend(a[0], a[1])
    raise NotImplementedError
