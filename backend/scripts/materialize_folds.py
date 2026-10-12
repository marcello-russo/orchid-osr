"""Materializza gli indici dei fold in un file versionato e auto-descrittivo.

Il training calcola i fold a runtime con StratifiedKFold(seed=57) sul contenuto
corrente del train dir: deterministico, ma implicito. Questo script fissa su
disco gli stessi indici (stesso codice, `split_raw_images`) per le 3
configurazioni della tesi x 2 size, cosi' un revisore puo' verificare e
riprodurre esattamente gli split:

    backend/data/kfold_indices_seed57.json   (versionato, ~1 MB)

Contenuto per configurazione: train/val come liste di path relativi
(`Classe/file`), conteggi per classe e totali, seed, md5 del manifest.
Il training usa il file quando la lista file corrente coincide, altrimenti
ricalcola a runtime con warning (vedi `split_raw_images`).

Uso:  cd backend && python scripts/materialize_folds.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.core.kfold.training import split_raw_images  # noqa: E402

SEED = 57
N_SPLITS = 5
SIZES = ["256x512", "224x224"]
CLASSES = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]
EXCLUDE_SETS = [[], ["O. garganica"], ["O. garganica", "O. majellensis"]]

OUT = BACKEND / "data" / "kfold_indices_seed57.json"


def key(size: str, exclude: list[str]) -> str:
    return f"{size}|{'full' if not exclude else '+'.join(sorted(exclude))}"


def main() -> None:
    manifest = BACKEND / "dataset" / "split_manifest.json"
    manifest_md5 = hashlib.md5(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    configs = {}
    for size in SIZES:
        train_dir = BACKEND / "dataset" / f"cropped_{size}" / "train"
        assert train_dir.is_dir(), f"manca {train_dir}"
        for exclude in EXCLUDE_SETS:
            paths, labels, folds = split_raw_images(
                str(train_dir), CLASSES, N_SPLITS, SEED,
                exclude_classes=exclude or None,
            )
            rel = [str(Path(p).relative_to(train_dir)) for p in paths]
            active = [c for c in CLASSES if c not in exclude]
            entry = {
                "size": size,
                "exclude_classes": sorted(exclude),
                "classes": active,
                "n_images": len(rel),
                "files": rel,
                "folds": [
                    {"train_idx": [int(i) for i in tr], "val_idx": [int(i) for i in va]}
                    for tr, va in folds
                ],
                "per_fold_counts": [
                    {
                        "train": dict(Counter(active[int(labels[i])] for i in tr)),
                        "val": dict(Counter(active[int(labels[i])] for i in va)),
                    }
                    for tr, va in folds
                ],
            }
            configs[key(size, exclude)] = entry
            print(f"{key(size, exclude)}: {len(rel)} img, {len(folds)} fold")
    OUT.write_text(json.dumps({
        "seed": SEED,
        "n_splits": N_SPLITS,
        "method": "sklearn StratifiedKFold(shuffle=True, random_state=57) su listing ordinato",
        "created": datetime.now(timezone.utc).isoformat(),
        "split_manifest_md5": manifest_md5,
        "configs": configs,
    }, indent=1), encoding="utf-8")
    print(f"scritto {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
