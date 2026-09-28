"""Agent 单次运行门面和统一错误边界。"""

import asyncio
from collections.abc import AsyncIterator

from tonyhar.resilience import ExecutionError, RunDeadlineExceeded
from tonyhar.tooling import ToolManager

from .agent_loop import AgentLoop
from .dispatcher import RunDispatcher, RunKind
from .events import build_event
from .llm import BaseLLM
from .run_handlers import (
    AgentHandler,
    DirectToolHandler,
    RetrievalAgentHandler,
    RunHandler,
)
from .runnables import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    AgentRunContext,
    AgentState,
    AgentStateMachine,
)
from .tool_runner import ToolRunner


class Agent(AgentStateMachine):
    """无会话状态的 Agent 运行门面。

    ``stream`` 是唯一执行入口。它只建立统一运行边界、请求路由和终态错误处理；
    三种运行策略由 Handler 承担，模型与 Tool 的循环由 ``AgentLoop`` 承担。
    所有跨请求会话状态仍由 ``ChatService`` 和 ``SessionStore`` 管理。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tool_manager: ToolManager,
        run_dispatcher: RunDispatcher,
        max_steps: int = 10,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        if not isinstance(tool_manager, ToolManager):
            raise TypeError("tool_manager 必须是 ToolManager")
        if not isinstance(run_dispatcher, RunDispatcher):
            raise TypeError("run_dispatcher 必须是 RunDispatcher")

        self.llm = llm
        self.tool_manager = tool_manager
        self.run_dispatcher = run_dispatcher
        self.max_steps = max_steps

        tool_runner = ToolRunner(tool_manager)
        agent_loop = AgentLoop(
            llm=llm,
            tool_manager=tool_manager,
            tool_runner=tool_runner,
            max_steps=max_steps,
        )
        self._handlers: dict[RunKind, RunHandler] = {
            RunKind.DIRECT_TOOL: DirectToolHandler(tool_runner),
            RunKind.RETRIEVAL_AGENT: RetrievalAgentHandler(
                tool_runner,
                agent_loop,
            ),
            RunKind.AGENT: AgentHandler(agent_loop),
        }

    async def stream(
        self,
        context: AgentRunContext,
    ) -> AsyncIterator[AgentEvent]:
        """路由一次独立运行，并转发 Handler 产生的标准事件。"""
        self._validate_context(context)
        yield build_event(
            context,
            AgentEventType.RUN_STARTED,
            state=context.state.value,
        )

        try:
            context.transition_to(AgentState.DISPATCHING)
            command = self.run_dispatcher.dispatch(context.user_input)
            context.add_message("user", context.user_input)
            yield build_event(
                context,
                AgentEventType.RUN_DISPATCHED,
                route_name=command.route_name,
                mode=command.kind.value,
                confidence=command.confidence,
                tool_calls=[
                    request.to_openai()
                    for request in command.tool_requests
                ],
            )

            handler = self._handlers[command.kind]
            async for event in handler.stream(context, command):
                yield event
        except asyncio.CancelledError:
            if context.state not in {
                AgentState.COMPLETED,
                AgentState.FAILED,
                AgentState.CANCELLED,
            }:
                context.transition_to(AgentState.CANCELLED)
            raise
        except ExecutionError as exc:
            if context.state not in {
                AgentState.COMPLETED,
                AgentState.FAILED,
                AgentState.CANCELLED,
            }:
                context.transition_to(AgentState.FAILED)
            yield build_event(
                context,
                AgentEventType.RUN_FAILED,
                success=False,
                stop_reason=(
                    "timeout"
                    if isinstance(exc, RunDeadlineExceeded)
                    else "dependency_error"
                ),
                steps=context.step,
                error_code=exc.error_code,
                error_message=str(exc),
            )
        except Exception as exc:
            if context.state not in {
                AgentState.COMPLETED,
                AgentState.FAILED,
                AgentState.CANCELLED,
            }:
                context.transition_to(AgentState.FAILED)
            yield build_event(
                context,
                AgentEventType.RUN_FAILED,
                success=False,
                stop_reason="error",
                steps=context.step,
                error_code="agent_runtime_error",
                error_message=str(exc),
            )

    async def invoke(self, context: AgentRunContext) -> AgentResult:
        """收集 ``stream`` 的终态，提供非流式结果适配。"""
        terminal_event: AgentEvent | None = None
        async for event in self.stream(context):
            if event.type in {
                AgentEventType.FINAL_ANSWER,
                AgentEventType.RUN_FAILED,
            }:
                terminal_event = event

        if terminal_event is None:
            return AgentResult(
                run_id=context.run_id,
                session_id=context.session_id,
                success=False,
                stop_reason="missing_terminal_event",
                steps=context.step,
                error_code="agent_missing_terminal_event",
                error_message="Agent 未产生终态事件",
            )
        return self._result_from_event(terminal_event)

    async def aclose(self) -> None:
        """释放模型客户端持有的连接池。"""
        await self.llm.aclose()

    @staticmethod
    def _validate_context(context: AgentRunContext) -> None:
        if not isinstance(context, AgentRunContext):
            raise TypeError("context 必须是 AgentRunContext")
        if context.state is not AgentState.CREATED:
            raise ValueError("AgentRunContext 必须从 created 状态开始")

    @staticmethod
    def _result_from_event(event: AgentEvent) -> AgentResult:
        data = event.data
        return AgentResult(
            run_id=event.run_id,
            session_id=event.session_id,
            success=bool(data.get("success", False)),
            answer=str(data.get("answer", "")),
            stop_reason=str(data.get("stop_reason", "unknown")),
            steps=int(data.get("steps", 0)),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
        )
