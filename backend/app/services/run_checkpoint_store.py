"""
SQLite-backed run ledger and stage checkpoints for recovery / audit trail.

Set MIROFISH_RUN_DB in .env to a writable SQLite file path. When unset, writes
are skipped (debug log only) so core APIs still run.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _db_path() -> Optional[str]:
    p = (os.environ.get("MIROFISH_RUN_DB") or "").strip()
    return p or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    if not path:
        raise RuntimeError("MIROFISH_RUN_DB is not set")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            project_id TEXT NOT NULL,
            simulation_id TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id);
        CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
        CREATE INDEX IF NOT EXISTS idx_runs_simulation ON runs(simulation_id);
        """
    )
    conn.commit()


def enabled() -> bool:
    return _db_path() is not None


def _project_run_key(project_id: str) -> str:
    return f"p-{project_id}"


def _sim_run_key(simulation_id: str) -> str:
    return f"s-{simulation_id}"


def _upsert_run(
    conn: sqlite3.Connection,
    run_id: str,
    kind: str,
    project_id: str,
    simulation_id: Optional[str],
    status: str = "active",
) -> None:
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO runs (run_id, kind, project_id, simulation_id, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            status = excluded.status,
            simulation_id = COALESCE(excluded.simulation_id, runs.simulation_id),
            updated_at = excluded.updated_at
        """,
        (run_id, kind, project_id, simulation_id, status, now, now),
    )


def checkpoint_project_stage(
    project_id: str,
    stage: str,
    payload: Dict[str, Any],
    *,
    simulation_id: Optional[str] = None,
    error: Optional[str] = None,
    kind: str = "project",
) -> Optional[int]:
    """
    Persist a checkpoint for a project-scoped run row (deterministic key p-{project_id}).
    """
    run_id = _project_run_key(project_id)
    if not enabled():
        logger.debug(
            "MIROFISH_RUN_DB unset; skipping project checkpoint stage=%s project=%s",
            stage,
            project_id,
        )
        return None
    now = _utc_now()
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            _upsert_run(
                conn, run_id, kind, project_id, simulation_id,
                status="failed" if error else "active",
            )
            conn.execute(
                """
                INSERT INTO checkpoints (run_id, stage, payload_json, error, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, stage, json.dumps(payload, ensure_ascii=False), error, now),
            )
            conn.commit()
            cur = conn.execute("SELECT last_insert_rowid()")
            return int(cur.fetchone()[0])
        finally:
            conn.close()


def checkpoint_simulation_stage(
    simulation_id: str,
    project_id: str,
    stage: str,
    payload: Dict[str, Any],
    error: Optional[str] = None,
) -> Optional[int]:
    """
    Persist a checkpoint for simulation preparation / runner (deterministic key s-{simulation_id}).
    """
    run_id = _sim_run_key(simulation_id)
    if not enabled():
        logger.debug(
            "MIROFISH_RUN_DB unset; skipping simulation checkpoint stage=%s sim=%s",
            stage,
            simulation_id,
        )
        return None
    now = _utc_now()
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            _upsert_run(
                conn,
                run_id,
                "simulation",
                project_id,
                simulation_id,
                status="failed" if error else "active",
            )
            conn.execute(
                """
                INSERT INTO checkpoints (run_id, stage, payload_json, error, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    stage,
                    json.dumps(payload, ensure_ascii=False),
                    error,
                    now,
                ),
            )
            conn.commit()
            cur = conn.execute("SELECT last_insert_rowid()")
            return int(cur.fetchone()[0])
        finally:
            conn.close()


def _latest_checkpoint(run_id: str, stage: str) -> Optional[Dict[str, Any]]:
    if not enabled():
        return None
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            cur = conn.execute(
                """
                SELECT payload_json, created_at
                FROM checkpoints
                WHERE run_id = ? AND stage = ? AND error IS NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id, stage),
            )
            row = cur.fetchone()
            if not row:
                return None
            payload_json, created_at = row
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid checkpoint JSON for run_id=%s stage=%s",
                    run_id,
                    stage,
                )
                return None
            return {"payload": payload, "created_at": created_at}
        finally:
            conn.close()


def load_project_stage_checkpoint(project_id: str, stage: str) -> Optional[Dict[str, Any]]:
    """Return the latest successful project-stage checkpoint payload and timestamp."""
    return _latest_checkpoint(_project_run_key(project_id), stage)


def load_simulation_stage_checkpoint(
    simulation_id: str,
    stage: str,
) -> Optional[Dict[str, Any]]:
    """Return the latest successful simulation-stage checkpoint payload and timestamp."""
    return _latest_checkpoint(_sim_run_key(simulation_id), stage)
