"""SQLite storage for uploaded recordings and the model-version history."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT, label TEXT, path TEXT, duration_s REAL,
            status TEXT DEFAULT 'pending',          -- pending | used
            created_at TEXT, used_in_version INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, trigger TEXT,
            n_new_recordings INTEGER, n_new_clips INTEGER,
            old_auc REAL, new_auc REAL, old_recall REAL, new_recall REAL,
            promoted INTEGER, note TEXT)""")


def add_upload(filename, label, path, duration_s):
    with connect() as c:
        cur = c.execute(
            "INSERT INTO uploads (filename, label, path, duration_s, created_at) VALUES (?,?,?,?,?)",
            (filename, label, str(path), duration_s, _now()))
        return cur.lastrowid


def pending_uploads():
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM uploads WHERE status='pending' ORDER BY id")]


def upload_counts():
    with connect() as c:
        rows = c.execute("SELECT status, label, COUNT(*) n FROM uploads GROUP BY status, label").fetchall()
    out = {"pending": {"normal": 0, "abnormal": 0}, "used": {"normal": 0, "abnormal": 0}}
    for r in rows:
        out.setdefault(r["status"], {"normal": 0, "abnormal": 0})[r["label"]] = r["n"]
    return out


def mark_used(upload_id, new_path, version_id):
    with connect() as c:
        c.execute("UPDATE uploads SET status='used', path=?, used_in_version=? WHERE id=?",
                  (str(new_path), version_id, upload_id))


def add_version(**kw):
    with connect() as c:
        cur = c.execute(
            """INSERT INTO model_versions (created_at, trigger, n_new_recordings, n_new_clips,
               old_auc, new_auc, old_recall, new_recall, promoted, note) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (_now(), kw.get("trigger"), kw.get("n_new_recordings"), kw.get("n_new_clips"),
             kw.get("old_auc"), kw.get("new_auc"), kw.get("old_recall"), kw.get("new_recall"),
             int(kw.get("promoted", False)), kw.get("note")))
        return cur.lastrowid


def list_versions():
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM model_versions ORDER BY id DESC")]
