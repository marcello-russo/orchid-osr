#!/usr/bin/env python3
"""Rotazione dei segreti in `secrets/app.env` (produzione: /srv/orchid/secrets/app.env).

Nato da un incidente vero: per ispezionare il container ho stampato l'ambiente
(`docker inspect --format '{{json .Config.Env}}'`) e `HF_TOKEN`, `JWT_SECRET` e
`SUPERADMIN_PASSWORD` sono finiti in un transcript. Ruotarli a mano è il tipo di
operazione che si sbaglia una volta su tre (riga duplicata, file troncato, mode
che cambia), quindi sta qui, con i test.

    scripts/rotate-app-secrets.py --env-file /srv/orchid/secrets/app.env                  # jwt + superadmin
    scripts/rotate-app-secrets.py --env-file ... --rotate jwt --dry-run
    scripts/rotate-app-secrets.py --env-file ... --rotate hf      # spiega perché non si può da qui

Cosa cambia:

- `JWT_SECRET` → 64 caratteri hex (256 bit, HS256). **Invalida tutte le sessioni
  in corso**: chi era loggato deve rifare il login, i guest pass già emessi
  smettono di valere. È l'unico modo di buttare fuori un token compromesso.
- `SUPERADMIN_PASSWORD` → 24 caratteri alfanumerici (`secrets.choice`). Non ha
  effetto finché il servizio non riparte: `ensure_superadmin_exists()` allinea
  l'hash nel DB all'avvio. Senza restart il file e il DB divergono e il login
  vecchio continua a funzionare.
- `HF_TOKEN` **non** è ruotabile qui: il token nuovo si crea sull'account
  HuggingFace (Settings → Access Tokens, scope *write* sui repo dataset/modelli)
  e poi si incolla nel file. Lo script lo rifiuta invece di inventare un valore
  che sembra valido e non lo è.

Il file non lascia mai il server in stato intermedio: si scrive su un file
temporaneo nella stessa directory, si applicano mode e owner originali, e solo
allora si sostituisce (`os.replace`, atomico). Qualsiasi errore rimuove il
temporaneo e lascia l'originale intatto.

Non stampa mai i valori: solo quali chiavi sono cambiate e quanto sono lunghe.
"""

from __future__ import annotations

import argparse
import os
import secrets
import string
import sys
import tempfile
from pathlib import Path

DEFAULT_ENV_FILE = "/srv/orchid/secrets/app.env"
JWT_HEX_CHARS = 64  # 32 byte = 256 bit
PASSWORD_CHARS = 24
PASSWORD_ALPHABET = string.ascii_letters + string.digits


def read_pairs(path: Path) -> list[tuple[str | None, str]]:
    """Righe `CHIAVE=valore` in ordine; commenti e righe vuote hanno chiave None."""
    pairs: list[tuple[str | None, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            pairs.append((None, line))
            continue
        key, _, value = line.partition("=")
        pairs.append((key.strip(), line))
    return pairs


def count_key(path: Path, key: str) -> int:
    return sum(1 for k, _ in read_pairs(path) if k == key)


def generate(key: str) -> str:
    if key == "JWT_SECRET":
        return secrets.token_hex(JWT_HEX_CHARS // 2)
    if key == "SUPERADMIN_PASSWORD":
        return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(PASSWORD_CHARS))
    raise SystemExit(f"errore: {key} non è generabile in locale")


def rewrite(path: Path, values: dict[str, str], dry_run: bool) -> None:
    lines = []
    for key, line in read_pairs(path):
        if key in values:
            lines.append(f"{key}={values[key]}\n")
        else:
            lines.append(line)

    if dry_run:
        print(f"dry-run: avrei riscritto {path} ({len(values)} chiavi)")
        return

    st = path.stat()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".app.env.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, st.st_mode & 0o777)
        if os.geteuid() == 0:
            os.chown(tmp, st.st_uid, st.st_gid)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ruota i segreti di app.env (jwt, superadmin).")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help=f"file da modificare (default {DEFAULT_ENV_FILE})")
    parser.add_argument(
        "--rotate",
        default="jwt,superadmin",
        help="csv tra jwt, superadmin, hf (default: jwt,superadmin)",
    )
    parser.add_argument("--dry-run", action="store_true", help="mostra cosa farebbe, non scrive")
    args = parser.parse_args(argv)

    path = Path(args.env_file)
    if not path.is_file():
        print(f"errore: {path} non esiste", file=sys.stderr)
        return 2

    requested = [part.strip().lower() for part in args.rotate.split(",") if part.strip()]
    known = {"jwt": "JWT_SECRET", "superadmin": "SUPERADMIN_PASSWORD"}
    if "hf" in requested or "hf_token" in requested:
        print(
            "errore: HF_TOKEN non si genera qui: va creato sull'account HuggingFace\n"
            "  (Settings -> Access Tokens, scope write su marcellorusso/orchid-osr-dataset\n"
            "  e -models), poi incollato nel file e applicato con un redeploy.",
            file=sys.stderr,
        )
        return 2
    unknown = [item for item in requested if item not in known]
    if unknown or not requested:
        print(f"errore: --rotate accetta {sorted(known)} (ricevuto {unknown or 'niente'})", file=sys.stderr)
        return 2

    # Il file è la fonte di verità del servizio: una chiave assente o duplicata
    # significa che qualcuno l'ha scritto a mano, e sovrascrivere alla cieca
    # produrrebbe un file con due righe e un comportamento dipendente dall'ordine.
    for key in (known[item] for item in requested):
        found = count_key(path, key)
        if found != 1:
            print(f"errore: {key} compare {found} volte in {path} (attesa: 1)", file=sys.stderr)
            return 2

    values = {known[item]: generate(known[item]) for item in requested}
    rewrite(path, values, args.dry_run)

    for item in requested:
        key = known[item]
        print(f"ruotato: {key} ({len(values[key])} caratteri)")
    print(f"invariato: il resto di {path}")
    if "jwt" in requested:
        print("nota: JWT_SECRET nuovo = tutte le sessioni e i guest pass in corso sono invalidati")
    if "superadmin" in requested:
        print("nota: SUPERADMIN_PASSWORD entra in vigore al riavvio del servizio (ensure_superadmin_exists)")
    if not args.dry_run:
        print("fatto. Riavvia con un deploy (scripts/deploy-vps.sh <tag>) perché il servizio rilegga il file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
