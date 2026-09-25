"""会话存储抽象及进程内实现。"""

import asyncio
from typing import Protocol

from .models import Session


class SessionStore(Protocol):
    async def create(self, session: Session) -> None:
        ...

    async def get(self, session_id: str) -> Session | None:
        ...

    async def save(self, session: Session) -> None:
        ...


class InMemorySessionStore:
    """适用于 CLI、单进程开发和单元测试的会话存储。"""

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
            self._sessions[session.session_id] = session.clone()
