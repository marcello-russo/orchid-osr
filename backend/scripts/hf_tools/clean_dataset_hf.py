#!/usr/bin/env python3
"""Allinea il dataset HF alla versione locale pulita (2192/300).

Dopo la rimozione locale dei 40 duplicati (vedi dedup_train_224.py):
  - elimina da HF i file orfani sotto cropped/224x224/train/ (attesi: 40);
  - carica split_manifest.json e hf_dataset_readme.md aggiornati.

Uso:
    python scripts/hf_tools/clean_dataset_hf.py            # dry-run
    python scripts/hf_tools/clean_dataset_hf.py --apply    # applica
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
DATASET_DIR = BACKEND_DIR / "dataset"
sys.path.insert(0, str(BACKEND_DIR))

from scripts.hf_config import HF_DATASET_REPO  # noqa: E402

_env = PROJECT_DIR / ".env"
if _env.exists():
    load_dotenv(_env)
TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    print("HF_TOKEN non trovato. Abort.", flush=True)
    sys.exit(1)

ALL_SPECIES = [
    "O. exaltata", "O. garganica", "O. incubacea", "O. majellensis",
    "O. sphegodes", "O. sphegodes_Palena",
]


def retry_call(fn, *args, attempts=20, wait=60, **kwargs):
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait_s = wait
                if "retry" in str(e).lower():
                    for w in str(e).split():
                        if w.isdigit() and int(w) > 0:
                            wait_s = max(wait, int(w))
                            break
                print(f"  rate limit ({i+1}/{attempts}): attendo {wait_s}s", flush=True)
                time.sleep(wait_s)
            else:
                print(f"  ERRORE non-rate-limit: {e}", flush=True)
                raise e
    raise RuntimeError(f"Fallito dopo {attempts} tentativi per rate limit")


def local_train224_set():
    """Set dei path relativi (cropped/224x224/train/...) locali canonici."""
    root = DATASET_DIR / "cropped_224x224" / "train"
    s = set()
    for sp in ALL_SPECIES:
        d = root / sp
        if not d.is_dir():
            continue
        for fn in os.listdir(d):
            s.add(f"cropped/224x224/train/{sp}/{fn}")
    return s


def main() -> int:
    apply = "--apply" in sys.argv
    api = HfApi(token=TOKEN)
    local = local_train224_set()
    print(f"Locali (cropped/224x224/train): {len(local)} file")

    hf_files = retry_call(api.list_repo_files, HF_DATASET_REPO, repo_type="dataset", token=TOKEN)
    hf_train224 = sorted(f for f in hf_files if f.startswith("cropped/224x224/train/"))
    print(f"HF (cropped/224x224/train): {len(hf_train224)} file")

    orphans = sorted(f for f in hf_train224 if f not in local)
    missing = sorted(f for f in local if f not in set(hf_train224))
    print(f"Orfan su HF (da eliminare): {len(orphans)}")
    print(f"Mancanti su HF (da caricare): {len(missing)}")
    for f in orphans:
        print(f"  DEL {f}")
    for f in missing:
        print(f"  ADD {f}")

    if len(orphans) != 40:
        print(f"ERRORE: attesi 40 orfani, trovati {len(orphans)}. Abort.")
        return 1
    if missing:
        print(f"ERRORE: {len(missing)} file locali assenti su HF. Abort.")
        return 1

    if not apply:
        print("\nDRY-RUN: nessuna modifica. Usa --apply per procedere.")
        return 0

    # 1) Elimina i 40 orfani
    ops = [CommitOperationDelete(path_in_repo=f) for f in orphans]
    retry_call(
        api.create_commit,
        HF_DATASET_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message="fix: rimossi 40 duplicati MD5 dal train 224x224 (uniformazione a 2192/300)",
        token=TOKEN,
    )
    print(f"  ✓ eliminati {len(ops)} file orfani")

    # 2) Carica manifest e readme aggiornati
    for name in ["split_manifest.json", "hf_dataset_readme.md"]:
        src = DATASET_DIR / name if name == "split_manifest.json" else PROJECT_DIR / "backend" / "scripts" / "hf_tools" / name
        if not src.exists():
            print(f"  WARN: {src} non trovato, saltato")
            continue
        retry_call(
            api.upload_file,
            path_or_fileobj=str(src),
            path_in_repo=name,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=TOKEN,
        )
        print(f"  ✓ uploaded {name}")

    # 3) Verifica finale
    hf_files = retry_call(api.list_repo_files, HF_DATASET_REPO, repo_type="dataset", token=TOKEN)
    n = len([f for f in hf_files if f.startswith("cropped/224x224/train/")])
    print(f"\nVerifica finale HF cropped/224x224/train: {n} file")
    assert n == 2192, f"Attesi 2192, trovati {n}"
    print("OK: dataset HF allineato a 2192/300.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
