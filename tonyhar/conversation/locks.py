"""会话粒度的异步互斥。"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


class SessionLockManager:
    """同一会话串行，不同会话使用不同的锁。"""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def lock(self, session_id: str) -> AsyncGenerator[None, None]:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            yield
