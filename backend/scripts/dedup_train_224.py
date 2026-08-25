#!/usr/bin/env python3
"""Rimuove i duplicati intra-train dalla pipeline 224x224 per uniformare i conteggi.

Contesto (verificato in sessione 25/08/2026):
  - train 256x512 = 2192 file (canonico)
  - train 224x224 = 2232 file, di cui 40 duplicati esatti (MD5) nelle 4 specie
    a fonti miste (exaltata +11, garganica +7, incubacea +14, sphegodes +8).
  - I 40 file "extra" hanno TUTTI un gemello MD5 identico gia' presente nel
    train 256x512 (file canonico). Rimuoverli non perde alcuna foto unica.

Invariant salvaguardati:
  - nessun file "extra" esiste nel train 256x512 (by filename);
  - ogni file "extra" ha almeno un gemello MD5 nel train 224x224 il cui
    filename E' presente nel train 256x512 (canonico);
  - dopo la rimozione: train 224x224 = 2192 file, tutti con MD5 unico.

Usage:
    python scripts/dedup_train_224.py            # dry-run (stampa il report)
    python scripts/dedup_train_224.py --apply    # elimina i 40 duplicati
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

DATASET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "dataset")
)
MIXED_SPECIES = ["O. exaltata", "O. garganica", "O. incubacea", "O. sphegodes"]
ALL_SPECIES = MIXED_SPECIES + ["O. majellensis", "O. sphegodes_Palena"]


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(root: str) -> dict:
    """Return {species: {filename}} for the train dirs under `root`."""
    result = {}
    for sp in ALL_SPECIES:
        d = os.path.join(root, sp)
        result[sp] = set(os.listdir(d)) if os.path.isdir(d) else set()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Elimina realmente i duplicati (default: dry-run)",
    )
    args = ap.parse_args()

    base = DATASET_DIR
    train224 = os.path.join(base, "cropped_224x224", "train")
    train256 = os.path.join(base, "cropped_256x512", "train")
    manifest_path = os.path.join(base, "split_manifest.json")

    # 1) File "extra" = presenti in 224 ma non in 256 (per filename)
    f224 = list_files(train224)
    f256 = list_files(train256)
    extras = []  # (species, filename)
    for sp in MIXED_SPECIES:
        for fn in sorted(f224[sp] - f256[sp]):
            extras.append((sp, fn))

    print(f"File extra (224\\256): {len(extras)}")
    if len(extras) != 40:
        print(f"ERRORE: attesi 40 extra, trovati {len(extras)}. Abort.")
        return 1

    # 2) Verifica: ogni extra ha gemello MD5 canonico nel train 224x224
    md5_index = defaultdict(list)  # md5 -> [(sp, fn)]
    for sp in ALL_SPECIES:
        d = os.path.join(train224, sp)
        for fn in f224[sp]:
            md5_index[md5(os.path.join(d, fn))].append((sp, fn))

    canon_ok = True
    report = []
    for sp, fn in extras:
        h = md5(os.path.join(train224, sp, fn))
        twins = [t for t in md5_index[h]]
        other_twins = [t for t in twins if (t[0], t[1]) != (sp, fn)]
        canonical_twins = [
            t for t in other_twins if t[1] in f256.get(t[0], set())
        ]
        if not other_twins:
            print(f"  ERRORE: {sp}/{fn} non ha gemelli MD5 in 224. Abort.")
            canon_ok = False
        if not canonical_twins:
            print(f"  ERRORE: {sp}/{fn} gemelli tutti extra (nessun canonico). Abort.")
            canon_ok = False
        report.append(
            (sp, fn, h[:8], ", ".join(f"{s}/{f}" for s, f in other_twins))
        )
    if not canon_ok:
        return 1

    # 3) Gli extra non devono comparire nel train 256
    for sp, fn in extras:
        assert fn not in f256[sp], f"{sp}/{fn} presente anche in 256"

    # 4) Report + rimozione
    print("\nReport (extra -> gemello/i MD5):")
    for sp, fn, h, twins in report:
        print(f"  {sp}/{fn}  [{h}]  ->  {twins}")

    before = sum(len(v) for v in f224.values())
    distinct_before = len(
        {md5(os.path.join(train224, sp, fn))
         for sp in ALL_SPECIES for fn in f224[sp]}
    )
    print(f"\nPrima: {before} file, {distinct_before} MD5 unici")

    if not args.apply:
        print("\nDRY-RUN: nessun file eliminato. Usa --apply per procedere.")
        return 0

    for sp, fn in extras:
        os.remove(os.path.join(train224, sp, fn))
        print(f"  rimosso {sp}/{fn}")

    f224 = list_files(train224)
    after = sum(len(v) for v in f224.values())
    distinct_after = len(
        {md5(os.path.join(train224, sp, fn))
         for sp in ALL_SPECIES for fn in f224[sp]}
    )
    print(f"Dopo: {after} file, {distinct_after} MD5 unici")
    assert after == 2192, f"Attesi 2192 file, trovati {after}"
    assert distinct_after == 2192, f"Attesi 2192 MD5 unici, trovati {distinct_after}"

    # 5) Aggiorna split_manifest.json (totali/percentuali pipeline 224)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["total_train_224x224"] = after
        manifest["per_class_train_224x224"] = dict(
            manifest.get("per_class_train_256x512", {})
        )
        manifest["note"] = (
            "sphegodes_Napoli rimossa dallo studio; train 224x224 deduplicato "
            "(40 duplicati MD5 rimossi) per uniformare i conteggi a 2192/300 "
            "in tutta la tesi."
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"split_manifest.json aggiornato (total_train_224x224 = {after})")
    else:
        print("WARN: split_manifest.json non trovato, saltato.")

    print("\nOK: train 224x224 uniformato a 2192/300.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
