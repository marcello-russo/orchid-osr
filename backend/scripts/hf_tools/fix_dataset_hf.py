#!/usr/bin/env python3
"""Aggiorna il dataset su HF alla versione locale corrente.

- Carica cropped/{224,256}x{512,224}/train+test corretti (versioni agosto)
- Elimina i file sphegodes_Napoli residui dal test e da raw/private
- Aggiorna split_manifest.json

Usa upload_large_folder per minimizzare i commit (1 per cartella) e
create_commit con DeleteOperation per eliminare i file in un unico commit.
Gestisce i rate limit HF (429) con retry automatico e wait.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from scripts.hf_config import HF_DATASET_REPO  # noqa: E402

_env = PROJECT_DIR / ".env"
if _env.exists():
    load_dotenv(_env)

TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    print("HF_TOKEN non trovato. Abort.", flush=True)
    sys.exit(1)

DATASET_DIR = BACKEND_DIR / "dataset"
LOG = PROJECT_DIR / "hf_dataset_fix.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def retry_call(fn, *args, attempts=20, wait=60, **kwargs):
    """Chiama fn con retry su 429 (rate limit), aspettando."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                # estrae il tempo di attesa se disponibile
                wait_s = wait
                if "retry" in str(e).lower():
                    for w in str(e).split():
                        if w.isdigit() and int(w) > 0:
                            wait_s = max(wait, int(w))
                            break
                log(f"  rate limit (tentativo {i+1}/{attempts}): attendo {wait_s}s")
                time.sleep(wait_s)
            else:
                log(f"  ERRORE non-rate-limit: {e}")
                raise e
    raise RuntimeError(f"Fallito dopo {attempts} tentativi per rate limit")


def upload_cropped():
    """Carica le 4 cartelle cropped corrette con upload_large_folder (1 commit)."""
    import shutil
    import tempfile

    staging = tempfile.mkdtemp(prefix="hf_fix_")
    try:
        for size, split in [("256x512", "train"), ("256x512", "test"), ("224x224", "train"), ("224x224", "test")]:
            local = DATASET_DIR / f"cropped_{size}" / split
            if not local.is_dir():
                log(f"SKIP cropped/{size}/{split}: {local} non esiste")
                continue
            dst = Path(staging) / "cropped" / size / split
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local, dst, dirs_exist_ok=True)
            log(f"  staged cropped/{size}/{split}")
        # manifest nello staging
        if (DATASET_DIR / "split_manifest.json").exists():
            shutil.copy2(DATASET_DIR / "split_manifest.json", Path(staging) / "split_manifest.json")
            log("  staged split_manifest.json")

        log("Upload con upload_large_folder (1 commit)...")
        api = HfApi(token=TOKEN)
        retry_call(
            api.upload_large_folder,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            folder_path=staging,
        )
        log("  ✓ cartelle cropped + manifest caricate")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def delete_napoli():
    """Elimina tutti i file sphegodes_Napoli in un unico commit."""
    api = HfApi(token=TOKEN)
    files = retry_call(api.list_repo_files, HF_DATASET_REPO, repo_type="dataset", token=TOKEN)
    napoli = [f for f in files if "sphegodes_Napoli" in f]
    if not napoli:
        log("  nessun file sphegodes_Napoli da eliminare")
        return
    log(f"Elimino {len(napoli)} file sphegodes_Napoli (un unico commit)...")
    ops = [CommitOperationDelete(path_in_repo=f) for f in napoli]
    retry_call(
        api.create_commit,
        HF_DATASET_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message="Remove sphegodes_Napoli (novel class rimossa dallo studio)",
        token=TOKEN,
    )
    log(f"  ✓ {len(napoli)} file sphegodes_Napoli eliminati")


def main():
    with open(LOG, "w") as f:
        f.write("")
    log("=== FIX DATASET HF: inizio ===")
    upload_cropped()
    delete_napoli()
    log("=== FIX DATASET HF: COMPLETATO ===")


if __name__ == "__main__":
    main()
