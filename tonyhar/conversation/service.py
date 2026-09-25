"""会话和 Agent 之间的应用服务。"""

from collections.abc import AsyncIterator
from uuid import uuid4

from tonyhar.resilience import Deadline
from tonyhar.agent.runnables import (
    AgentEvent,
    AgentResult,
    AgentRunContext,
    AgentStateMachine,
)

from .locks import SessionLockManager
from .models import Session
from .store import SessionStore


class ChatService:
    """负责会话加载、隔离、串行化和持久化。"""

    def __init__(
        self,
        agent: AgentStateMachine,
        sessions: SessionStore,
        *,
        system_prompt: str = "",
        max_turns: int = 20,
        run_timeout_seconds: float = 90.0,
        locks: SessionLockManager | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 必须大于 0")
        if run_timeout_seconds <= 0:
            raise ValueError("run_timeout_seconds 必须大于 0")
        self._agent = agent
        self._sessions = sessions
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._run_timeout_seconds = run_timeout_seconds
        self._locks = locks or SessionLockManager()

    async def create_session(self, session_id: str | None = None) -> Session:
        resolved_id = (session_id or uuid4().hex).strip()
        session = Session.create(
            resolved_id,
            system_prompt=self._system_prompt,
            max_turns=self._max_turns,
        )
        async with self._locks.lock(resolved_id):
            await self._sessions.create(session)
        return session.clone()

    async def invoke(
        self,
        session_id: str,
        user_input: str,
    ) -> AgentResult:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 不能为空")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input 不能为空")

        resolved_id = session_id.strip()
        async with self._locks.lock(resolved_id):
            session = await self._sessions.get(resolved_id)
            if session is None:
                session = Session.create(
                    resolved_id,
                    system_prompt=self._system_prompt,
                    max_turns=self._max_turns,
                )

            context = AgentRunContext(
                run_id=uuid4().hex,
                session_id=resolved_id,
                user_input=user_input.strip(),
                messages=session.snapshot(),
                deadline=Deadline.after(self._run_timeout_seconds),
            )
            try:
                return await self._agent.invoke(context)
            finally:
                session.replace_messages(context.messages)
                await self._sessions.save(session)

    async def stream(
        self,
        session_id: str,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """以事件流方式运行一轮会话。

        这个方法是 Web SSE、WebSocket 等实时接口的适配入口。它本身不
        实现 Agent 的推理、工具调用或状态转换，而是负责把“会话管理”和
        “Agent 运行时”连接起来：

        1. 校验并规范化输入，避免空会话 ID 或空消息进入运行时。
        2. 获取会话级异步锁。同一个 ``session_id`` 的请求必须串行，避免
           两次请求同时读取旧消息后互相覆盖；不同会话使用不同的锁，仍然
           可以并发执行。
        3. 从 ``SessionStore`` 读取会话。不存在的会话按当前系统提示词和
           窗口大小自动创建，这样首次发送消息不必先显式创建会话。
        4. 使用会话消息的快照创建 ``AgentRunContext``。Agent 只修改本次
           运行上下文，不直接修改存储中的会话，避免运行到一半时外部读到
           不完整的消息序列。
        5. 调用 ``Agent.stream(context)``，逐个转发
           ``AgentEvent``。调用方可以据此实时显示意图识别、模型调用、工具
           调用和最终答案等进度。
        6. 无论 Agent 正常完成、抛出异常，还是客户端在消费流时断开，
           ``finally`` 都会把上下文中的消息写回会话存储。锁也会在离开
           ``async with`` 后释放，因此不会永久阻塞后续请求。

        注意：这里的“流”是 Agent 事件流，不等同于模型逐 token 的文本
        流。若要实现逐字输出，需要在 ``BaseLLM`` 增加模型供应商的流式
        接口，并由 Agent 产生文本增量事件。
        """
        # 在取得锁之前完成输入校验，非法请求不会占用会话资源。
        self._validate_input(session_id, user_input)
        resolved_id = session_id.strip()

        # 锁的生命周期覆盖整个 async generator。只要调用方还在消费事件，
        # 当前会话就不会被另一条请求并发修改；不同 session_id 不共享这把锁。
        async with self._locks.lock(resolved_id):
            # SessionStore 返回的是独立副本，避免 Agent 直接修改存储对象。
            session = await self._sessions.get(resolved_id)
            if session is None:
                session = Session.create(
                    resolved_id,
                    system_prompt=self._system_prompt,
                    max_turns=self._max_turns,
                )

            # 用历史消息快照初始化本次运行。context.messages 会在 Agent 内部
            # 追加 user、assistant 和 tool 消息，运行完成后再统一保存。
            context = AgentRunContext(
                run_id=uuid4().hex,
                session_id=resolved_id,
                user_input=user_input.strip(),
                messages=session.snapshot(),
                deadline=Deadline.after(self._run_timeout_seconds),
            )
            try:
                # 不缓存事件、不等到 Agent 结束才返回；每收到一个事件就
                # 立即 yield 给 FastAPI SSE 或其他实时调用方。
                async for event in self._agent.stream(context):
                    yield event
            finally:
                # 事件流结束时提交本次上下文。即使 Agent 失败或客户端取消
                # 消费，已产生的消息也会被保存，保证会话状态可恢复。
                session.replace_messages(context.messages)
                await self._sessions.save(session)

    async def aclose(self) -> None:
        """关闭 Agent 持有的模型连接池。"""
        await self._agent.aclose()

    @staticmethod
    def _validate_input(session_id: str, user_input: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 不能为空")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input 不能为空")
