"""The asset store (DESIGN 3.6): content-hash ids, sidecars, and a SQLite index for search."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import uuid
import zipfile
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from audiokit import dsp, wav
from audiokit.wav import atomic_write_bytes

from .errors import BadRequest, NotFound, ValidationFailed
from .jobs import ASSET_ID_PATTERN

ASSET_ID_RE = re.compile(ASSET_ID_PATTERN)
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
ALLOWED_EXTENSIONS = (".wav", ".mid", ".zip", ".json")
MAX_SEARCH_LIMIT = 200
MAX_NOTE_CHARS = 2000
MAX_ZIP_ENTRIES = 20_000
MAX_ZIP_UNCOMPRESSED = 2 * 1024**3
_HASH_CHUNK = 1024 * 1024

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY, job_id TEXT, type TEXT NOT NULL, kind TEXT NOT NULL,
    model TEXT, family TEXT, prompt TEXT NOT NULL DEFAULT '', tags_json TEXT NOT NULL DEFAULT '{}',
    relpath TEXT NOT NULL, size_bytes INTEGER NOT NULL, rating INTEGER, note TEXT,
    created_at TEXT NOT NULL, last_used_at TEXT
);
CREATE TABLE IF NOT EXISTS asset_tags (
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    key TEXT NOT NULL COLLATE NOCASE, value TEXT NOT NULL COLLATE NOCASE,
    PRIMARY KEY (asset_id, key, value)
);
CREATE INDEX IF NOT EXISTS assets_created ON assets(created_at);
CREATE INDEX IF NOT EXISTS assets_job ON assets(job_id);
CREATE INDEX IF NOT EXISTS tags_kv ON asset_tags(key, value);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TIME_FORMAT)


def parse_time(value: str | datetime) -> datetime:
    """Parse an ISO 8601 time (naive means UTC) or raise :class:`ValidationFailed`."""
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except (ValueError, AttributeError) as exc:
            raise ValidationFailed([{"path": "since", "message": "must be an ISO 8601 time"}]) from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    """Recursively make ``value`` JSON-serialisable: non-finite floats become ``None``, numpy scalars plain numbers."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        return json_safe(item())
    return str(value)


def file_asset_id(path: Path) -> str:
    """``a_`` + the first 24 hex characters of the SHA-256 of the file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return "a_" + digest.hexdigest()[:24]


def check_asset_id(asset_id: Any) -> str:
    if not isinstance(asset_id, str) or not ASSET_ID_RE.match(asset_id):
        raise NotFound("no such asset", code="asset_not_found")
    return asset_id


class AssetStore:
    """Files under ``data_dir/assets/<yyyy>/<mm>/<id>.<ext>`` with a ``<id>.json`` sidecar each."""

    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.root = self.data_dir / "assets"
        self.uploads_dir = self.data_dir / "uploads"
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.data_dir / "index.sqlite", check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- writing ----

    def add_asset(self, src: Path, sidecar: Mapping[str, Any], *, move: bool = True) -> dict[str, Any]:
        """Store ``src`` under its content-hash id and write the sidecar. If the same bytes are already
        stored, the existing asset (and its sidecar) wins and ``src`` is discarded (``move``)."""
        ext = src.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise BadRequest(f"unsupported asset type {ext!r}")
        asset_id = file_asset_id(src)
        created = parse_time(str(sidecar.get("created_at") or utc_now()))
        with self._lock:
            existing = self._row(asset_id)
            if existing is not None:
                if move:
                    src.unlink(missing_ok=True)
                return self.meta(asset_id)
            rel = Path(f"{created:%Y}") / f"{created:%m}" / f"{asset_id}{ext}"
            dest = self.root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".tmp-{asset_id}{ext}")
            if move:
                shutil.move(str(src), tmp)
            else:
                shutil.copyfile(src, tmp)
            os.replace(tmp, dest)
            full = dict(json_safe(dict(sidecar)))
            full.update(
                asset_id=asset_id,
                created_at=created.strftime(TIME_FORMAT),
                size_bytes=dest.stat().st_size,
                ext=ext,
            )
            full.setdefault("rating", None)
            full.setdefault("note", None)
            atomic_write_bytes(dest.with_suffix(".json"), json.dumps(full, indent=2, allow_nan=False).encode("utf-8"))
            self._index(full, rel.as_posix())
            return full

    def _index(self, side: Mapping[str, Any], relpath: str) -> None:
        tags: dict[str, Any] = side["tags"] if isinstance(side.get("tags"), dict) else {}
        self._db.execute("BEGIN")
        try:
            self._db.execute(
                "INSERT INTO assets(asset_id, job_id, type, kind, model, family, prompt, tags_json, relpath,"
                " size_bytes, rating, note, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    side["asset_id"], side.get("job_id"), side.get("type") or "", side.get("kind") or "",
                    side.get("model"), side.get("family"), side.get("prompt") or "", json.dumps(tags),
                    relpath, side["size_bytes"], side.get("rating"), side.get("note"), side["created_at"],
                ),
            )
            for key, value in tags.items():
                for item in [value] if isinstance(value, str) else value:
                    self._db.execute(
                        "INSERT OR IGNORE INTO asset_tags(asset_id, key, value) VALUES (?,?,?)",
                        (side["asset_id"], str(key), str(item)),
                    )
            self._db.execute("COMMIT")
        except sqlite3.Error:
            self._db.execute("ROLLBACK")
            raise

    def register_upload(
        self, src: Path, *, original_name: str | None = None, max_bytes: int, max_seconds: float
    ) -> dict[str, Any]:
        """Validate an uploaded file (a WAV, or a zip holding one ``.scraft``) and store it as an asset."""
        try:
            size = src.stat().st_size
        except OSError as exc:
            raise BadRequest("upload not readable", code="invalid_upload") from exc
        if size > max_bytes:
            raise BadRequest(f"upload larger than {max_bytes} bytes", code="upload_too_large", status=413)
        with open(src, "rb") as handle:
            magic = handle.read(12)
        name = (original_name or "")[:255]
        side: dict[str, Any] = {
            "job_id": None, "type": "upload", "model": None, "family": None, "checkpoint": None,
            "checkpoint_hash": None, "licence": None, "prompt": "", "tags": {}, "seed": None, "params": {},
            "requested": {}, "parents": [], "candidate": 0, "original_name": name,
        }
        if magic[:4] == b"RIFF" and magic[8:12] == b"WAVE":
            audio, sr = self._decode_wav(src, max_seconds)
            side.update(
                kind="audio", sample_rate=sr, channels=int(audio.shape[0]), bit_depth=None,
                measured={
                    "duration_s": round(audio.shape[1] / sr, 4), "peak_dbfs": round(dsp.peak_dbfs(audio), 2),
                    "rms_dbfs": round(dsp.rms_dbfs(audio), 2),
                },
            )
            staged = self._stage(src, ".wav")
        elif magic[:2] == b"PK":
            self._check_session_zip(src)
            side.update(kind="session", sample_rate=None, channels=None, bit_depth=None, measured={})
            staged = self._stage(src, ".zip")
        else:
            raise BadRequest("upload must be a WAV file or a zip of a session", code="invalid_upload")
        return self.add_asset(staged, side)

    def _stage(self, src: Path, ext: str) -> Path:
        staged = self.uploads_dir / f".stage-{uuid.uuid4().hex}{ext}"
        shutil.copyfile(src, staged)
        return staged

    @staticmethod
    def _decode_wav(path: Path, max_seconds: float) -> tuple[Any, int]:
        try:
            audio, sr = wav.read(path)
        except (wav.WavError, OSError, ValueError) as exc:
            raise BadRequest(f"not a readable WAV file: {exc}", code="invalid_upload") from exc
        if audio.shape[1] == 0:
            raise BadRequest("WAV file contains no audio", code="invalid_upload")
        if audio.shape[1] / sr > max_seconds:
            raise BadRequest(f"audio longer than {max_seconds:g} s", code="upload_too_long", status=413)
        return audio, sr

    @staticmethod
    def _check_session_zip(path: Path) -> None:
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_ZIP_ENTRIES or sum(i.file_size for i in infos) > MAX_ZIP_UNCOMPRESSED:
                    raise BadRequest("session archive too large", code="invalid_upload")
                for info in infos:
                    parts = Path(info.filename.replace("\\", "/")).parts
                    if info.filename.startswith(("/", "\\")) or ".." in parts or ":" in (parts[0] if parts else ""):
                        raise BadRequest("session archive has an unsafe path", code="invalid_upload")
                scraft = [i for i in infos if not i.is_dir() and i.filename.lower().endswith(".scraft")]
        except zipfile.BadZipFile as exc:
            raise BadRequest("not a valid zip file", code="invalid_upload") from exc
        if len(scraft) != 1:
            raise BadRequest("a session archive must contain exactly one .scraft file", code="invalid_upload")

    # ---- reading ----

    def _row(self, asset_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()

    def exists(self, asset_id: str) -> bool:
        return ASSET_ID_RE.match(asset_id) is not None and self._row(asset_id) is not None

    def _require(self, asset_id: str) -> sqlite3.Row:
        row = self._row(check_asset_id(asset_id))
        if row is None:
            raise NotFound("no such asset", code="asset_not_found")
        return row

    def path(self, asset_id: str) -> Path:
        """Absolute path of the stored file."""
        return self.root / str(self._require(asset_id)["relpath"])

    def kind(self, asset_id: str) -> str:
        return str(self._require(asset_id)["kind"])

    def meta(self, asset_id: str) -> dict[str, Any]:
        """The sidecar."""
        path = self.path(asset_id).with_suffix(".json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise NotFound("sidecar missing", code="asset_not_found") from exc
        return data if isinstance(data, dict) else {}

    def search(
        self,
        *,
        tags: Mapping[str, str | list[str]] | None = None,
        text: str | None = None,
        type: str | None = None,
        model: str | None = None,
        job_id: str | None = None,
        rated: bool | None = None,
        min_rating: int | None = None,
        since: str | datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Assets matching every given filter, newest first (``limit`` is clamped to 1..200)."""
        where: list[str] = ["1=1"]
        args: list[Any] = []
        for key, wanted in (tags or {}).items():
            values = [wanted] if isinstance(wanted, str) else list(wanted)
            if not values:
                continue
            marks = ",".join("?" * len(values))
            where.append(f"EXISTS (SELECT 1 FROM asset_tags t WHERE t.asset_id = assets.asset_id AND t.key = ? AND t.value IN ({marks}))")
            args += [key, *values]
        if text:
            escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("prompt LIKE ? ESCAPE '\\' COLLATE NOCASE")
            args.append(f"%{escaped}%")
        for column, value in (("type", type), ("model", model), ("job_id", job_id)):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        if rated is not None:
            where.append("rating IS NOT NULL" if rated else "rating IS NULL")
        if min_rating is not None:
            where.append("rating >= ?")
            args.append(int(min_rating))
        if since is not None:
            where.append("created_at >= ?")
            args.append(parse_time(since).strftime(TIME_FORMAT))
        limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
        sql = f"SELECT * FROM assets WHERE {' AND '.join(where)} ORDER BY created_at DESC, rowid DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, [*args, limit]).fetchall()
        return [self._summary(r) for r in rows]

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "asset_id": row["asset_id"], "job_id": row["job_id"], "type": row["type"], "kind": row["kind"],
            "model": row["model"], "prompt": row["prompt"], "tags": json.loads(row["tags_json"]),
            "rating": row["rating"], "note": row["note"], "size_bytes": row["size_bytes"],
            "created_at": row["created_at"],
        }

    # ---- mutation ----

    def rate(self, asset_id: str, rating: int, note: str | None = None) -> dict[str, Any]:
        """Record a 0..5 verdict (and optional note); returns the updated sidecar."""
        if isinstance(rating, bool) or not isinstance(rating, int) or not 0 <= rating <= 5:
            raise ValidationFailed([{"path": "rating", "message": "must be an integer from 0 to 5"}])
        if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE_CHARS):
            raise ValidationFailed([{"path": "note", "message": f"must be text of at most {MAX_NOTE_CHARS} characters"}])
        with self._lock:
            side = self.meta(asset_id)
            side["rating"] = rating
            if note is not None:
                side["note"] = note
            atomic_write_bytes(self.path(asset_id).with_suffix(".json"), json.dumps(side, indent=2, allow_nan=False).encode("utf-8"))
            self._db.execute("UPDATE assets SET rating = ?, note = COALESCE(?, note) WHERE asset_id = ?", (rating, note, asset_id))
            return side

    def mark_used(self, asset_id: str) -> None:
        """Note that the asset was used as a job input (it then outlives the retention window)."""
        with self._lock:
            self._db.execute("UPDATE assets SET last_used_at = ? WHERE asset_id = ?", (utc_now(), asset_id))

    def purge(self, retention_days: int, now: datetime | None = None) -> list[str]:
        """Delete unrated assets created, and last used, more than ``retention_days`` ago."""
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=retention_days)).strftime(TIME_FORMAT)
        with self._lock:
            rows = self._db.execute(
                "SELECT asset_id, relpath FROM assets WHERE rating IS NULL AND created_at < ?"
                " AND (last_used_at IS NULL OR last_used_at < ?)",
                (cutoff, cutoff),
            ).fetchall()
            removed = []
            for row in rows:
                data = self.root / row["relpath"]
                data.unlink(missing_ok=True)
                data.with_suffix(".json").unlink(missing_ok=True)
                self._db.execute("DELETE FROM assets WHERE asset_id = ?", (row["asset_id"],))
                removed.append(row["asset_id"])
            return removed

    def count(self) -> int:
        with self._lock, closing(self._db.execute("SELECT COUNT(*) FROM assets")) as cur:
            return int(cur.fetchone()[0])
