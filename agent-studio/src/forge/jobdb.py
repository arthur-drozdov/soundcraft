"""SQLite persistence of jobs (DESIGN 3.5)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from .jobs import JobRecord, JobResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT UNIQUE NOT NULL, type TEXT NOT NULL,
    model TEXT, family TEXT, lane TEXT, state TEXT NOT NULL, priority TEXT NOT NULL,
    params_json TEXT NOT NULL, inputs_json TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '', error TEXT, result_json TEXT, idempotency_key TEXT,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_key ON jobs(idempotency_key, created_at);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
"""
MAX_LIST = 200


class JobDB:
    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        path = Path(data_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path / "jobs.sqlite", check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def insert(self, rec: JobRecord) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs(job_id, type, model, family, lane, state, priority, params_json, inputs_json,"
                " progress, message, error, result_json, idempotency_key, created_at, started_at, finished_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.job_id, rec.type, rec.model, rec.family, rec.lane, rec.state, rec.priority,
                 json.dumps(rec.params), json.dumps(rec.inputs), rec.progress, rec.message, rec.error,
                 self._result(rec), rec.idempotency_key, rec.created_at, rec.started_at, rec.finished_at),
            )

    @staticmethod
    def _result(rec: JobRecord) -> str | None:
        return None if rec.result is None else rec.result.model_dump_json()

    def save(self, rec: JobRecord) -> None:
        """Persist the mutable fields (state, progress, error, result, timestamps)."""
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET state=?, progress=?, message=?, error=?, result_json=?, started_at=?, finished_at=?"
                " WHERE job_id=?",
                (rec.state, rec.progress, rec.message, rec.error, self._result(rec), rec.started_at, rec.finished_at, rec.job_id),
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        result = row["result_json"]
        return JobRecord(
            job_id=row["job_id"], type=row["type"], model=row["model"], family=row["family"], lane=row["lane"],
            state=row["state"], priority=row["priority"], params=json.loads(row["params_json"]),
            inputs=json.loads(row["inputs_json"]), progress=row["progress"], message=row["message"],
            error=row["error"], result=JobResult.model_validate_json(result) if result else None,
            idempotency_key=row["idempotency_key"], created_at=row["created_at"],
            started_at=row["started_at"], finished_at=row["finished_at"],
        )

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else self._record(row)

    def list(self, state: str | None = None, limit: int = 50) -> list[JobRecord]:
        """Newest jobs first."""
        limit = max(1, min(int(limit), MAX_LIST))
        sql, args = "SELECT * FROM jobs", []
        if state is not None:
            sql += " WHERE state = ?"
            args.append(state)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY seq DESC LIMIT ?", [*args, limit]).fetchall()
        return [self._record(r) for r in rows]

    def find_by_key(self, key: str, since: str) -> JobRecord | None:
        """The newest job with this idempotency key created at or after ``since`` (ISO UTC)."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ? AND created_at >= ? ORDER BY seq DESC LIMIT 1",
                (key, since),
            ).fetchone()
        return None if row is None else self._record(row)

    def fail_unfinished(self, message: str, now: str) -> int:
        """Mark every queued or running job failed (used at startup); returns how many."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET state='failed', error=?, finished_at=? WHERE state IN ('queued','running')",
                (message, now),
            )
            return cur.rowcount
