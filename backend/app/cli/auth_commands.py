"""Comandi CLI di gestione utenti e inviti.

Estratto da `experiments.py`: qui vive solo l'adattamento argparse → API
interna (`app.core.auth`). Nessun accesso diretto a file o database, così CLI e
API condividono la stessa fonte di verità e le stesse validazioni.
"""

from __future__ import annotations

import argparse

from fastapi import HTTPException

from app.core.auth import (
    MAX_INVITE_HOURS,
    create_user,
    ensure_superadmin_exists,
    generate_invite,
    list_users_and_invites,
)


def _wants_auth_command(args: argparse.Namespace) -> bool:
    return any(
        [
            getattr(args, "list_users", False),
            getattr(args, "gen_invites", None) is not None,
            getattr(args, "add_user", None) is not None,
            getattr(args, "username", None) is not None,
        ]
    )


def handle_auth_commands(args: argparse.Namespace) -> int | None:
    """Esegue un comando di gestione utenti.

    Ritorna l'exit code, oppure None se `args` non chiede nulla di auth.
    """
    if not _wants_auth_command(args):
        return None

    ensure_superadmin_exists()

    if getattr(args, "list_users", False):
        data = list_users_and_invites()
        print("=== Users ===")
        for user in data["users"]:
            print(f"  {user['username']} ({user['role']})")
        print("\n=== Invites ===")
        for invite in data["invites"]:
            if invite["revoked"]:
                status = "revoked"
            elif invite["used"]:
                status = "used"
            else:
                status = "available"
            expires = f", expires {invite['expires_at']}" if invite.get("expires_at") else ""
            print(f"  {invite['code']} [{status}] ({invite['role']}{expires})")
        return 0

    if getattr(args, "gen_invites", None) is not None:
        hours = int(getattr(args, "invite_hours", 72))
        if hours > MAX_INVITE_HOURS:
            print(f"--invite-hours must be 0..{MAX_INVITE_HOURS} (0 = no expiry)")
            return 1
        try:
            codes = generate_invite(role="user", count=args.gen_invites, expires_in_hours=hours)
        except HTTPException as exc:
            print(exc.detail)
            return 1
        for code in codes:
            print(code)
        return 0

    add_user = getattr(args, "add_user", None)
    if add_user is not None or getattr(args, "username", None) is not None:
        password = getattr(args, "password", None)
        if not password:
            print("--password is required with --add-user")
            return 1
        username = add_user or getattr(args, "username", None)
        if not username:
            print("Specify username with --add-user <name> or --username")
            return 1
        role = getattr(args, "role", "user")
        try:
            create_user(username, password, role)
        except HTTPException as exc:
            if exc.status_code == 409:
                print(f"User '{username}' already exists")
            else:
                print(exc.detail)
            return 1
        print(f"User '{username}' created with role '{role}'")
        return 0

    return None
