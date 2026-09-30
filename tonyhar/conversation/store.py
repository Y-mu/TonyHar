"""会话存储端口、内存实现和 SQLite 实现。"""

import asyncio
import json
from pathlib import Path
import sqlite3
from typing import Protocol

from .models import Session


class SessionConflictError(RuntimeError):
    """会话修订版本冲突，防止并发覆盖。"""


class SessionStore(Protocol):
    async def create(self, session: Session) -> None:
        ...

    async def get(self, session_id: str) -> Session | None:
        ...

    async def save(self, session: Session) -> None:
        ...

    async def aclose(self) -> None:
        ...


class InMemorySessionStore:
    """适用于测试的进程内会话存储。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self, session: Session) -> None:
        async with self._lock:
            if session.session_id in self._sessions:
                raise ValueError(f"会话 '{session.session_id}' 已存在")
            self._sessions[session.session_id] = session.clone()

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            return session.clone() if session is not None else None

    async def save(self, session: Session) -> None:
        async with self._lock:
            current = self._sessions.get(session.session_id)
            if current is None:
                self._sessions[session.session_id] = session.clone()
                return
            if current.revision != session.revision:
                raise SessionConflictError(
                    f"会话 '{session.session_id}' 已被其他请求更新"
                )
            session.revision += 1
            self._sessions[session.session_id] = session.clone()

    async def aclose(self) -> None:
        return None


class SQLiteSessionStore:
    """单体部署使用的持久化会话存储。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()
        self._lock = asyncio.Lock()

    async def create(self, session: Session) -> None:
        async with self._lock:
            await asyncio.to_thread(self._create_sync, session)

    def _create_sync(self, session: Session) -> None:
        try:
            self._connection.execute(
                "INSERT INTO sessions(session_id, revision, payload) VALUES (?, ?, ?)",
                (
                    session.session_id,
                    session.revision,
                    json.dumps(session.to_dict(), ensure_ascii=False),
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"会话 '{session.session_id}' 已存在") from exc

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, session_id)

    def _get_sync(self, session_id: str) -> Session | None:
        row = self._connection.execute(
            "SELECT revision, payload FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[1])
        data["revision"] = int(row[0])
        return Session.from_dict(data)

    async def save(self, session: Session) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save_sync, session)

    def _save_sync(self, session: Session) -> None:
        row = self._connection.execute(
            "SELECT revision FROM sessions WHERE session_id = ?",
            (session.session_id,),
        ).fetchone()
        if row is None:
            self._create_sync(session)
            return
        if int(row[0]) != session.revision:
            raise SessionConflictError(
                f"会话 '{session.session_id}' 已被其他请求更新"
            )

        next_revision = session.revision + 1
        data = session.to_dict()
        data["revision"] = next_revision
        cursor = self._connection.execute(
            """
            UPDATE sessions
            SET revision = ?, payload = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ? AND revision = ?
            """,
            (
                next_revision,
                json.dumps(data, ensure_ascii=False),
                session.session_id,
                session.revision,
            ),
        )
        if cursor.rowcount != 1:
            self._connection.rollback()
            raise SessionConflictError(
                f"会话 '{session.session_id}' 已被其他请求更新"
            )
        self._connection.commit()
        session.revision = next_revision

    async def aclose(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._connection.close)
