"""SQLite database layer per metadati. Fallback a JSON per retrocompatibilita."""

from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import sqlite3
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Path assoluto: backend/data/orchid.db
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = str(_BACKEND_DIR / "data" / "orchid.db")


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model_variant TEXT NOT NULL,
    num_folds INTEGER DEFAULT 5,
    num_classes INTEGER NOT NULL,
    loss_type TEXT,
    phases TEXT,
    recipe TEXT,
    mean_f1 REAL,
    std_f1 REAL,
    mean_accuracy REAL,
    test_f1 REAL,
    test_accuracy REAL,
    best_fold INTEGER,
    status TEXT DEFAULT 'completed',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    mtime REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS folds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    fold_idx INTEGER NOT NULL CHECK(fold_idx >= 0),
    best_f1 REAL,
    best_accuracy REAL,
    best_epoch INTEGER,
    final_train_loss REAL,
    final_val_loss REAL,
    train_size INTEGER DEFAULT 0,
    val_size INTEGER DEFAULT 0,
    history TEXT,
    phase_epochs TEXT,
    UNIQUE(experiment_id, fold_idx)
);

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'admin')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invites (
    code TEXT PRIMARY KEY,
    used INTEGER NOT NULL DEFAULT 0 CHECK(used IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS login_attempts (
    username TEXT PRIMARY KEY,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_folds_experiment ON folds(experiment_id);
CREATE INDEX IF NOT EXISTS idx_folds_experiment_fold ON folds(experiment_id, fold_idx);
"""

# Colonne NCD aggiunte dopo la creazione iniziale della tabella
_NCD_COLUMNS = [
    ("ncd_auroc", "REAL"),
    ("ncd_aupr", "REAL"),
    ("ncd_detector", "TEXT"),
    ("ncd_clustering_nmi", "REAL"),
    ("is_ncd", "INTEGER DEFAULT 0"),
]


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Aggiunge colonne mancanti (migrazione backward-compatibile)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()}
    for col_name, col_type in _NCD_COLUMNS:
        if col_name not in existing:
            try:
                conn.execute(f"ALTER TABLE experiments ADD COLUMN {col_name} {col_type}")
                logger.info("Migrazione DB: aggiunta colonna %s", col_name)
            except sqlite3.OperationalError:
                logger.warning("Migrazione DB: impossibile aggiungere %s", col_name)
    conn.commit()


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    _migrate_schema(conn)


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("phases", "history", "phase_epochs"):
        if isinstance(d.get(key), str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _read_file(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_ncd_metrics(exp_dir: str) -> dict:
    """Legge le metriche NCD da ncd_results.json e calcola i valori migliori."""
    ncd_file = os.path.join(exp_dir, "ncd_results.json")
    if not os.path.exists(ncd_file):
        return {}
    ncd_data = _read_file(ncd_file)
    if not ncd_data:
        return {}

    metrics = {}

    detectors = ncd_data.get("detectors", {})
    if detectors and isinstance(detectors, dict):
        # Regola ufficiale (tesi): Ensemble, stesso rivelatore per tutti.
        ens = detectors.get("Ensemble")
        if isinstance(ens, dict) and isinstance(ens.get("auroc"), (int, float)):
            metrics["ncd_auroc"] = ens["auroc"]
            metrics["ncd_aupr"] = ens.get("aupr")
            metrics["ncd_detector"] = "Ensemble"
        else:
            best_auroc = -1.0
            best_aupr = -1.0
            best_detector = None
            for name, det_data in detectors.items():
                if not isinstance(det_data, dict):
                    continue
                auroc = det_data.get("auroc")
                if isinstance(auroc, (int, float)) and auroc > best_auroc:
                    best_auroc = auroc
                    best_aupr = det_data.get("aupr", best_aupr)
                    best_detector = name
            if best_auroc >= 0:
                metrics["ncd_auroc"] = best_auroc
                metrics["ncd_aupr"] = best_aupr if best_aupr >= 0 else None
                metrics["ncd_detector"] = best_detector

    clustering = ncd_data.get("clustering", {})
    if clustering and isinstance(clustering, dict):
        best_nmi = -1.0
        for name, cl_data in clustering.items():
            if not isinstance(cl_data, dict):
                continue
            nmi = cl_data.get("nmi", cl_data.get("novel_nmi"))
            if isinstance(nmi, (int, float)) and nmi > best_nmi:
                best_nmi = nmi
        if best_nmi >= 0:
            metrics["ncd_clustering_nmi"] = best_nmi

    return metrics


def sync_experiment(exp_dir: str, conn: sqlite3.Connection | None = None) -> bool:
    own_conn = conn is None
    if own_conn:
        conn = get_db()

    try:
        config_file = os.path.join(exp_dir, "config.json")
        results_file = os.path.join(exp_dir, "results.json")
        if not os.path.exists(config_file) and not os.path.exists(results_file):
            return False

        exp_id = os.path.basename(exp_dir)

        max_mtime = 0.0
        for f in [config_file, results_file]:
            if os.path.exists(f):
                m = os.path.getmtime(f)
                if m > max_mtime:
                    max_mtime = m

        existing = conn.execute("SELECT mtime FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        if existing and existing["mtime"] is not None and existing["mtime"] >= max_mtime:
            return False

        config = _read_file(config_file) or {}
        results = _read_file(results_file) or {}
        test_metrics = results.get("test_metrics") if isinstance(results.get("test_metrics"), dict) else {}

        started_at_raw = results.get("started_at")
        if started_at_raw is not None:
            started = started_at_raw
        else:
            started = datetime.datetime.fromtimestamp(
                min(os.path.getmtime(f) for f in [config_file, results_file] if os.path.exists(f))
            ).isoformat()
        completed = datetime.datetime.fromtimestamp(max_mtime).isoformat() if results else None

        has_ncd_file = os.path.exists(os.path.join(exp_dir, "ncd_results.json"))
        is_ncd = 1 if config.get("ncd_mode") else 0
        ncd_metrics = _read_ncd_metrics(exp_dir) if has_ncd_file else {}

        # Build column list dynamically per includere campi NCD
        base_cols = [
            "id",
            "name",
            "model_variant",
            "num_folds",
            "num_classes",
            "loss_type",
            "phases",
            "recipe",
            "mean_f1",
            "std_f1",
            "mean_accuracy",
            "test_f1",
            "test_accuracy",
            "best_fold",
            "status",
            "started_at",
            "completed_at",
            "mtime",
        ]
        ncd_cols = ["ncd_auroc", "ncd_aupr", "ncd_detector", "ncd_clustering_nmi", "is_ncd"]
        all_cols = base_cols + ncd_cols

        placeholders = ", ".join("?" for _ in all_cols)
        columns = ", ".join(all_cols)

        base_vals = [
            exp_id,
            config.get("name") or config.get("type") or exp_id,
            config.get("model_variant", "unknown"),
            results.get("num_folds") or config.get("num_folds", 5),
            config.get("num_classes", 0),
            config.get("loss_type"),
            json.dumps(config.get("phases")) if config.get("phases") else None,
            config.get("recipe"),
            results.get("mean_f1"),
            results.get("std_f1"),
            results.get("mean_accuracy"),
            test_metrics.get("f1_macro") or results.get("test_f1"),
            test_metrics.get("accuracy") or results.get("test_accuracy"),
            results.get("best_fold"),
            "completed" if results else "running",
            started,
            completed,
            max_mtime,
        ]
        ncd_vals = [
            ncd_metrics.get("ncd_auroc"),
            ncd_metrics.get("ncd_aupr"),
            ncd_metrics.get("ncd_detector"),
            ncd_metrics.get("ncd_clustering_nmi"),
            is_ncd,
        ]

        conn.execute(
            f"INSERT OR REPLACE INTO experiments ({columns}) VALUES ({placeholders})",
            base_vals + ncd_vals,
        )

        for ff in sorted(glob.glob(os.path.join(exp_dir, "fold_*.json"))):
            fold_data = _read_file(ff)
            if fold_data is None:
                continue
            fold_idx = int(os.path.basename(ff).replace("fold_", "").replace(".json", ""))

            history = fold_data.get("history")
            phase_epochs = fold_data.get("phase_epochs")

            conn.execute(
                """INSERT OR REPLACE INTO folds
                   (experiment_id, fold_idx, best_f1, best_accuracy, best_epoch,
                    final_train_loss, final_val_loss, train_size, val_size,
                    history, phase_epochs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    exp_id,
                    fold_idx,
                    fold_data.get("best_f1"),
                    fold_data.get("best_accuracy", fold_data.get("sphor_probe_accuracy")),
                    fold_data.get("best_epoch", 0),
                    fold_data.get("final_train_loss"),
                    fold_data.get("final_val_loss"),
                    fold_data.get("train_size", 0),
                    fold_data.get("val_size", 0),
                    json.dumps(history) if history else None,
                    json.dumps(phase_epochs) if phase_epochs else None,
                ),
            )

        conn.commit()
        return True
    finally:
        if own_conn:
            conn.close()


def sync_all() -> int:
    conn = get_db()
    synced = 0
    experiments_dir = os.path.abspath(settings.results_dir)
    for d in sorted(glob.glob(os.path.join(experiments_dir, "*", ""))):
        dirname = os.path.basename(d.rstrip("/"))
        if not dirname.startswith("exp"):
            continue
        if sync_experiment(d.rstrip("/"), conn):
            synced += 1
    conn.close()
    return synced
