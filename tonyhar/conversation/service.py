"""会话、短期记忆和 Agent 之间的应用服务。"""

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

from tonyhar.agent.runnables import (
    AgentEvent,
    AgentResult,
    AgentRunContext,
    AgentState,
    AgentStateMachine,
)
from tonyhar.resilience import Deadline

from .locks import SessionLockManager
from .memory import (
    ContextBuilder,
    ContextPolicy,
    ExtractiveMemoryCompactor,
    MemoryCompactor,
    TokenCounter,
)
from .models import Session, TurnStatus
from .store import SessionStore


class ChatService:
    """负责会话日志、短期记忆、隔离、串行化和持久化。"""

    def __init__(
        self,
        agent: AgentStateMachine,
        sessions: SessionStore,
        *,
        system_prompt: str = "",
        run_timeout_seconds: float = 90.0,
        context_builder: ContextBuilder | None = None,
        memory_compactor: MemoryCompactor | None = None,
        locks: SessionLockManager | None = None,
    ) -> None:
        if run_timeout_seconds <= 0:
            raise ValueError("run_timeout_seconds 必须大于 0")
        counter = TokenCounter()
        self._agent = agent
        self._sessions = sessions
        self._system_prompt = system_prompt
        self._run_timeout_seconds = run_timeout_seconds
        self._context_builder = context_builder or ContextBuilder(
            counter,
            ContextPolicy(),
        )
        self._memory_compactor = memory_compactor or ExtractiveMemoryCompactor(
            counter,
            self._context_builder.policy.summary_max_tokens,
        )
        self._locks = locks or SessionLockManager()

    async def create_session(self, session_id: str | None = None) -> Session:
        resolved_id = (session_id or uuid4().hex).strip()
        session = Session.create(
            resolved_id,
            system_prompt=self._system_prompt,
        )
        async with self._locks.lock(resolved_id):
            await self._sessions.create(session)
        return session.clone()

    async def get_session(self, session_id: str) -> Session | None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 不能为空")
        resolved_id = session_id.strip()
        async with self._locks.lock(resolved_id):
            session = await self._sessions.get(resolved_id)
        return session.clone() if session is not None else None

    async def invoke(self, session_id: str, user_input: str) -> AgentResult:
        self._validate_input(session_id, user_input)
        resolved_id = session_id.strip()
        normalized_input = user_input.strip()
        async with self._locks.lock(resolved_id):
            session = await self._load_or_create(resolved_id)
            deadline = Deadline.after(self._run_timeout_seconds)
            messages = await self._prepare_context(
                session,
                normalized_input,
                deadline,
            )
            context = AgentRunContext(
                run_id=uuid4().hex,
                session_id=resolved_id,
                user_input=normalized_input,
                messages=messages,
                deadline=deadline,
            )
            initial_message_count = len(context.messages)
            try:
                return await self._agent.invoke(context)
            except asyncio.CancelledError:
                self._mark_cancelled(context)
                raise
            finally:
                self._commit_run(session, context, initial_message_count)
                await self._sessions.save(session)

    async def stream(
        self,
        session_id: str,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        self._validate_input(session_id, user_input)
        resolved_id = session_id.strip()
        normalized_input = user_input.strip()
        async with self._locks.lock(resolved_id):
            session = await self._load_or_create(resolved_id)
            deadline = Deadline.after(self._run_timeout_seconds)
            messages = await self._prepare_context(
                session,
                normalized_input,
                deadline,
            )
            context = AgentRunContext(
                run_id=uuid4().hex,
                session_id=resolved_id,
                user_input=normalized_input,
                messages=messages,
                deadline=deadline,
            )
            initial_message_count = len(context.messages)
            try:
                async for event in self._agent.stream(context):
                    yield event
            except asyncio.CancelledError:
                self._mark_cancelled(context)
                raise
            finally:
                self._commit_run(session, context, initial_message_count)
                await self._sessions.save(session)

    async def _load_or_create(self, session_id: str) -> Session:
        session = await self._sessions.get(session_id)
        if session is not None:
            return session
        return Session.create(session_id, system_prompt=self._system_prompt)

    async def _prepare_context(
        self,
        session: Session,
        user_input: str,
        deadline: Deadline,
    ) -> list[dict]:
        policy = self._context_builder.policy
        for _ in range(policy.max_compactions_per_run):
            if not self._context_builder.needs_compaction(session, user_input):
                break
            candidates = self._context_builder.compaction_candidates(session)
            if not candidates:
                break
            session.memory = await self._memory_compactor.compact(
                session.memory,
                candidates,
                deadline=deadline,
            )
        return self._context_builder.build(session, user_input)

    @staticmethod
    def _commit_run(
        session: Session,
        context: AgentRunContext,
        initial_message_count: int,
    ) -> None:
        new_messages = context.messages[initial_message_count:]
        if not new_messages:
            return
        status = {
            AgentState.COMPLETED: TurnStatus.COMPLETED,
            AgentState.CANCELLED: TurnStatus.CANCELLED,
        }.get(context.state, TurnStatus.FAILED)
        session.append_turn(
            new_messages,
            status=status,
            turn_id=context.run_id,
        )

    @staticmethod
    def _mark_cancelled(context: AgentRunContext) -> None:
        if context.state not in {
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }:
            context.transition_to(AgentState.CANCELLED)

    async def aclose(self) -> None:
        try:
            await self._agent.aclose()
        finally:
            try:
                await self._memory_compactor.aclose()
            finally:
                await self._sessions.aclose()

    @staticmethod
    def _validate_input(session_id: str, user_input: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 不能为空")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input 不能为空")
