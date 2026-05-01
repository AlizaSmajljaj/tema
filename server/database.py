"""
database.py — SQLite database for user persistence.
Fixed for compatibility with older SQLite versions by using Python timestamps.
"""

import sqlite3
import secrets
import time
import json
import os
from pathlib import Path
from typing import Optional


def get_db_path():
    env_path = os.environ.get("DB_PATH")
    if env_path:
        return env_path
    if os.path.exists("/data"):
        return "/data/haskell_tutor.db"
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        return "/tmp/haskell_tutor.db"
    return str(Path(__file__).parent.parent / "haskell_tutor.db")

DB_PATH = get_db_path()

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

print(f"[DB] Using database at: {DB_PATH}", flush=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    NOT NULL UNIQUE,
            token      TEXT    NOT NULL UNIQUE,
            created_at REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS problem_sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            problem_id TEXT    NOT NULL,
            code       TEXT    NOT NULL DEFAULT '',
            solved     INTEGER NOT NULL DEFAULT 0,
            updated_at REAL    NOT NULL,
            UNIQUE(user_id, problem_id)
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            problem_id TEXT    NOT NULL,
            context    TEXT    NOT NULL DEFAULT 'error',
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            timestamp  REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS experience (
            user_id   INTEGER NOT NULL REFERENCES users(id),
            category  TEXT    NOT NULL,
            encounters INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(user_id, category)
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_user    ON problem_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_convos_user_prob ON conversations(user_id, problem_id);
        CREATE INDEX IF NOT EXISTS idx_experience_user  ON experience(user_id);
        """)
    print(f"[DB] Tables initialised.", flush=True)



def register_user(username: str) -> Optional[dict]:
    token = secrets.token_urlsafe(32)
    now = time.time()
    print(f"[DB] Registering user: '{username}'", flush=True)
    try:
        conn = get_conn()
        cur = conn.execute(
            "INSERT INTO users (username, token, created_at) VALUES (?, ?, ?)",
            (username.strip().lower(), token, now)
        )
        conn.close()
        print(f"[DB] Registered successfully, id={cur.lastrowid}", flush=True)
        return {"id": cur.lastrowid, "username": username.strip().lower(), "token": token}
    except sqlite3.IntegrityError:
        print(f"[DB] Username already taken: '{username}'", flush=True)
        return None
    except Exception as e:
        print(f"[DB] Unexpected error in register_user: {e}", flush=True)
        return None

def get_user_by_token(token: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, username, token FROM users WHERE token = ?", (token,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_username(username: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, username, token FROM users WHERE username = ?",
        (username.strip().lower(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None



def save_code(user_id: int, problem_id: str, code: str):
    now = time.time()
    conn = get_conn()
    conn.execute("""
        INSERT INTO problem_sessions (user_id, problem_id, code, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, problem_id) DO UPDATE SET
            code = excluded.code,
            updated_at = excluded.updated_at
    """, (user_id, problem_id, code, now))
    conn.close()

def mark_problem_solved(user_id: int, problem_id: str):
    now = time.time()
    conn = get_conn()
    row = conn.execute(
        "SELECT id FROM problem_sessions WHERE user_id = ? AND problem_id = ?",
        (user_id, problem_id)
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE problem_sessions SET solved = 1, updated_at = ? WHERE user_id = ? AND problem_id = ?",
            (now, user_id, problem_id)
        )
    else:
        conn.execute(
            "INSERT INTO problem_sessions (user_id, problem_id, solved, updated_at) VALUES (?, ?, 1, ?)",
            (user_id, problem_id, now)
        )
    conn.close()

def get_user_progress(user_id: int) -> dict:
    conn = get_conn()
    rows = conn.execute(
        "SELECT problem_id, code, solved, updated_at FROM problem_sessions WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    conn.close()
    return {
        "solved": [r["problem_id"] for r in rows if r["solved"]],
        "in_progress": {r["problem_id"]: r["code"] for r in rows if not r["solved"] and r["code"]},
        "total_solved": sum(1 for r in rows if r["solved"]),
    }

def get_saved_code(user_id: int, problem_id: str) -> str:
    conn = get_conn()
    row = conn.execute(
        "SELECT code FROM problem_sessions WHERE user_id = ? AND problem_id = ?",
        (user_id, problem_id)
    ).fetchone()
    conn.close()
    return row["code"] if row else ""



def save_message(user_id: int, problem_id: str, role: str, content: str, context: str = "error"):
    now = time.time()
    conn = get_conn()
    conn.execute(
        "INSERT INTO conversations (user_id, problem_id, context, role, content, timestamp) VALUES (?,?,?,?,?,?)",
        (user_id, problem_id, context, role, content, now)
    )
    conn.close()

def get_conversation(user_id: int, problem_id: str, context: str = "error", limit: int = 20) -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT role, content FROM conversations
        WHERE user_id = ? AND problem_id = ? AND context = ?
        ORDER BY timestamp DESC LIMIT ?
    """, (user_id, problem_id, context, limit)).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def clear_conversation(user_id: int, problem_id: str, context: str = "error"):
    conn = get_conn()
    conn.execute(
        "DELETE FROM conversations WHERE user_id = ? AND problem_id = ? AND context = ?",
        (user_id, problem_id, context)
    )
    conn.close()



def increment_category(user_id: int, category: str):
    conn = get_conn()
    conn.execute("""
        INSERT INTO experience (user_id, category, encounters)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, category) DO UPDATE SET
            encounters = encounters + 1
    """, (user_id, category))
    conn.close()

def get_experience(user_id: int) -> dict:
    conn = get_conn()
    rows = conn.execute(
        "SELECT category, encounters FROM experience WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    conn.close()
    return {r["category"]: r["encounters"] for r in rows}



def get_user_stats(user_id: int) -> dict:
    progress = get_user_progress(user_id)
    exp = get_experience(user_id)
    conn = get_conn()
    msg_count = conn.execute(
        "SELECT COUNT(*) as c FROM conversations WHERE user_id = ?", (user_id,)
    ).fetchone()["c"]
    conn.close()
    return {
        "solved": progress["total_solved"],
        "in_progress": len(progress["in_progress"]),
        "messages_sent": msg_count,
        "experience": exp,
    }

init_db()