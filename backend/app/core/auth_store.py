"""Persistenza di utenti, inviti e passi ospite: unica fonte di verità.

Tutte le scritture passano da `transaction()` (BEGIN IMMEDIATE): due richieste
concorrenti sullo stesso invito monouso vengono serializzate da SQLite, non
dalla buona volontà del chiamante. Le letture usano una connessione breve e
non tengono lock.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.db import get_db

logger = logging.getLogger(__name__)

# Quanto vivono gli inviti importati dal JSON, che non avevano scadenza.
# Uguale a `app.core.auth.MAX_INVITE_HOURS`: importarlo qui creerebbe un ciclo
# (auth importa auth_store), quindi il valore è replicato e verificato dal test
# `test_legacy_invite_ttl_matches_api_policy`.
_LEGACY_INVITE_HOURS = 720

_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def reset_for_tests() -> None:
    """Dimentica l'import legacy (i test cambiano DB tra un caso e l'altro)."""
    global _INITIALIZED
    _INITIALIZED = False


def initialize(legacy_path: Path) -> None:
    """Importa users.json una volta sola, poi marca il DB. Idempotente."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        with transaction() as conn:
            _import_legacy_json(conn, legacy_path)
        _INITIALIZED = True


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    with _connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _normalize_invite(row: dict[str, Any]) -> dict[str, Any]:
    row["used"] = bool(row["used"])
    row["revoked"] = bool(row["revoked"])
    return row


def _normalize_guest(row: dict[str, Any]) -> dict[str, Any]:
    row["revoked"] = bool(row["revoked"])
    return row


# ── utenti ──────────────────────────────────────────────────────────────────


def get_user(username: str) -> dict[str, Any] | None:
    with _connection() as conn:
        return _row_dict(conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone())


def find_user(conn: sqlite3.Connection, username: str) -> dict[str, Any] | None:
    return _row_dict(conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone())


def list_users() -> list[dict[str, Any]]:
    with _connection() as conn:
        return [dict(r) for r in conn.execute("SELECT username, role FROM users ORDER BY username")]


def insert_user(conn: sqlite3.Connection, username: str, password_hash: str, role: str) -> None:
    conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (username, password_hash, role, datetime.now(UTC).isoformat()),
    )


def upsert_user(conn: sqlite3.Connection, username: str, password_hash: str, role: str) -> None:
    conn.execute(
        """
        INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET password_hash = excluded.password_hash, role = excluded.role
        """,
        (username, password_hash, role, datetime.now(UTC).isoformat()),
    )


def delete_user(conn: sqlite3.Connection, username: str) -> None:
    conn.execute("DELETE FROM users WHERE username = ?", (username,))


# ── inviti ──────────────────────────────────────────────────────────────────


def get_invite(code: str) -> dict[str, Any] | None:
    with _connection() as conn:
        return find_invite(conn, code)


def find_invite(conn: sqlite3.Connection, code: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM invites WHERE code = ?", (code,)).fetchone()
    return _normalize_invite(dict(row)) if row is not None else None


def insert_invite(conn: sqlite3.Connection, invite: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO invites (code, role, used, used_by, created_at, expires_at, max_uses, uses, label, revoked)
        VALUES (:code, :role, :used, :used_by, :created_at, :expires_at, :max_uses, :uses, :label, :revoked)
        """,
        {**invite, "used": int(bool(invite["used"])), "revoked": int(bool(invite["revoked"]))},
    )


def consume_invite(conn: sqlite3.Connection, code: str, username: str) -> None:
    row = conn.execute("SELECT uses, max_uses FROM invites WHERE code = ?", (code,)).fetchone()
    if row is None:
        raise KeyError(f"invito inesistente: {code}")
    uses = int(row["uses"]) + 1
    conn.execute(
        "UPDATE invites SET uses = ?, used = ?, used_by = ? WHERE code = ?",
        (uses, 1 if uses >= int(row["max_uses"]) else 0, username, code),
    )


def set_invite_revoked(conn: sqlite3.Connection, code: str) -> bool:
    cur = conn.execute("UPDATE invites SET revoked = 1 WHERE code = ?", (code,))
    return cur.rowcount > 0


def list_invites() -> list[dict[str, Any]]:
    with _connection() as conn:
        return [_normalize_invite(dict(r)) for r in conn.execute("SELECT * FROM invites ORDER BY created_at")]


# ── passi ospite ────────────────────────────────────────────────────────────


def get_guest_pass(jti: str) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute("SELECT * FROM guest_passes WHERE jti = ?", (jti,)).fetchone()
        return _normalize_guest(dict(row)) if row is not None else None


def insert_guest_pass(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO guest_passes (jti, label, created_at, expires_at, revoked) VALUES (?, ?, ?, ?, 0)",
        (record["jti"], record["label"], record["created_at"], record["expires_at"]),
    )


def set_guest_pass_revoked(conn: sqlite3.Connection, jti: str) -> bool:
    cur = conn.execute("UPDATE guest_passes SET revoked = 1 WHERE jti = ?", (jti,))
    return cur.rowcount > 0


# ── import legacy ───────────────────────────────────────────────────────────


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO auth_meta (key, value) VALUES (?, ?)", (key, value))


def _as_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"users.json: '{key}' deve essere una lista di oggetti")
    return value


def _import_legacy_json(conn: sqlite3.Connection, path: Path) -> None:
    if conn.execute("SELECT 1 FROM auth_meta WHERE key = 'legacy_json_imported'").fetchone():
        return
    if not path.exists():
        # NON si marca niente: "file assente" non è "migrazione fatta". Marcarlo
        # disattiverebbe l'import per sempre, e il rimedio ovvio (rimettere il
        # file e riavviare) non importerebbe niente, in silenzio.
        logger.info("users.json assente in %s: nessun import legacy, si riprova al prossimo avvio", path)
        return

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("users.json: l'oggetto radice deve essere un dizionario")

    users = _as_list(raw, "users")
    invites = _as_list(raw, "invites")
    passes = _as_list(raw, "guest_passes")
    now = datetime.now(UTC).isoformat()

    for user in users:
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (user["username"], user["password_hash"], user.get("role", "user"), user.get("created_at") or now),
        )
    # Un invito legacy non ha scadenza: resterebbe spendibile per sempre, contro
    # l'invariante che l'API ora impone. Agli inviti ancora utilizzabili se ne
    # dà una (stesso cap dell'API); quelli già consumati non servono a nessuno.
    legacy_deadline = (datetime.now(UTC) + timedelta(hours=_LEGACY_INVITE_HOURS)).isoformat()
    for invite in invites:
        used = bool(invite.get("used"))
        expires_at = invite.get("expires_at") or (None if used else legacy_deadline)
        conn.execute(
            """
            INSERT OR IGNORE INTO invites
                (code, role, used, used_by, created_at, expires_at, max_uses, uses, label, revoked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invite["code"],
                invite.get("role", "user"),
                int(used),
                invite.get("used_by"),
                invite.get("created_at") or now,
                expires_at,
                int(invite.get("max_uses", 1)),
                int(invite.get("uses", 1 if used else 0)),
                invite.get("label", ""),
                int(bool(invite.get("revoked"))),
            ),
        )
    for record in passes:
        conn.execute(
            "INSERT OR IGNORE INTO guest_passes (jti, label, created_at, expires_at, revoked) VALUES (?, ?, ?, ?, ?)",
            (
                record["jti"],
                record.get("label", ""),
                record.get("created_at") or now,
                record.get("expires_at") or now,
                int(bool(record.get("revoked"))),
            ),
        )

    _set_meta(conn, "legacy_json_imported", now)
    logger.info(
        "Import legacy users.json: %d utenti, %d inviti, %d passi (file lasciato come rollback)",
        len(users),
        len(invites),
        len(passes),
    )
