"""Agent 状态机运行时。"""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

from tonyhar.resilience import ExecutionError, RunDeadlineExceeded
from tonyhar.tooling import ToolExecutor, ToolRegistry, ToolRequest, ToolResult

from .intent_planner import ExecutionMode, IntentPlanner
from .llm import BaseLLM, LLMResponse
from .runnables import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    AgentRunContext,
    AgentState,
    AgentStateMachine,
)


class Agent(AgentStateMachine):
    """无会话状态的 Agent 状态机。

    ``Agent`` 只保存可复用的基础设施依赖：LLM、工具注册表、意图规划器
    和最大步数；它不保存任何用户的消息、当前状态或上一次运行结果。因此，
    同一个 Agent 实例可以被多个会话并发使用。

    单次运行的全部可变数据都在 ``AgentRunContext`` 中，状态转换如下：

    ```text
    CREATED
      ↓
    PLANNING
      ├── 直接工具 ──→ EXECUTING_TOOLS ──→ COMPLETED / FAILED
      └── Agent 推理 ─→ CALLING_MODEL
                            ├── 最终文本 ──→ COMPLETED
                            └── tool_calls → EXECUTING_TOOLS
                                               └── CALLING_MODEL（循环）

    任意运行中状态 ──取消──→ CANCELLED
    任意运行中状态 ──异常──→ FAILED
    ```

    ``stream`` 是唯一真正执行入口；``invoke`` 只是收集终态，方便 CLI
    或非流式调用方使用。调用方必须为每一轮创建新的
    ``AgentRunContext``，不能复用已经进入终态的 context。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        intent_planner: IntentPlanner,
        max_steps: int = 10,
        max_tool_concurrency: int = 4,
    ):
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        self.llm = llm
        self.tool_executor = ToolExecutor(
            tools,
            max_concurrency=max_tool_concurrency,
        )
        self.intent_planner = intent_planner
        self.max_steps = max_steps

    async def stream(
        self,
        context: AgentRunContext,
    ) -> AsyncIterator[AgentEvent]:
        """执行一次独立运行，并按状态机推进顺序产生事件。

        这里的“无状态”不是说运行过程中没有状态，而是说状态不存放在
        ``Agent`` 对象上。每次调用都遵循同一套流程：

        1. 校验 context 必须处于 ``CREATED``，然后发出运行开始事件。
        2. 进入 ``PLANNING``，由 ``IntentPlanner`` 生成本轮执行计划。
        3. 如果计划包含确定性工具调用，进入 ``EXECUTING_TOOLS``；工具结果
           写回 context，并根据结果直接完成，或继续交给模型。
        4. 需要模型推理时进入 ``CALLING_MODEL``，调用 LLM。
        5. 模型返回普通文本时进入 ``COMPLETED``；模型返回 tool calls 时先
           记录 assistant tool-call 消息，再进入 ``EXECUTING_TOOLS``。
        6. 工具完成后回到 ``CALLING_MODEL``，如此循环直到得到最终文本或达到
           ``max_steps``。
        7. 取消请求进入 ``CANCELLED``；未处理异常进入 ``FAILED``。这两种
           终态都不会被伪装成成功答案。

        所有状态转换都通过 ``AgentRunContext.transition_to`` 校验，事件只
        反映当前 context，不修改 Agent 的共享字段。这样不同会话可以共享
        一个 Agent 实例，而不会互相覆盖状态。
        """
        self._validate_context(context)
        yield self._event(
            context,
            AgentEventType.RUN_STARTED,
            state=context.state.value,
        )

        try:
            context.transition_to(AgentState.PLANNING)
            plan = self.intent_planner.plan(context.user_input)
            context.add_message("user", plan.user_message)
            yield self._event(
                context,
                AgentEventType.INTENT_PLANNED,
                route_name=plan.route_name,
                mode=plan.mode.value,
                confidence=plan.confidence,
                tool_calls=[request.to_openai() for request in plan.tool_calls],
            )

            planned_results: list[ToolResult] = []
            if plan.tool_calls:
                context.transition_to(AgentState.EXECUTING_TOOLS)
                context.pending_tools = list(plan.tool_calls)
                async for event in self._execute_tools_stream(
                    context,
                    plan.tool_calls,
                    record_request=True,
                ):
                    yield event
                planned_results = list(context.last_tool_results)
                context.pending_tools.clear()

            if plan.mode is ExecutionMode.DIRECT_TOOL:
                answer = self._format_direct_results(planned_results)
                context.add_message("assistant", answer)
                failed_result = next(
                    (result for result in planned_results if not result.success),
                    None,
                )
                if failed_result is not None:
                    context.transition_to(AgentState.FAILED)
                    yield self._event(
                        context,
                        AgentEventType.RUN_FAILED,
                        success=False,
                        answer=answer,
                        stop_reason="tool_error",
                        steps=context.step,
                        error_code=failed_result.error_code,
                        error_message=failed_result.error_message,
                    )
                    return

                context.transition_to(AgentState.COMPLETED)
                yield self._event(
                    context,
                    AgentEventType.FINAL_ANSWER,
                    success=True,
                    answer=answer,
                    stop_reason="completed",
                    steps=context.step,
                )
                return

            context.transition_to(AgentState.CALLING_MODEL)
            for step in range(1, self.max_steps + 1):
                context.step = step
                yield self._event(
                    context,
                    AgentEventType.MODEL_STARTED,
                    step=step,
                )
                response = await self.llm.chat(
                    messages=context.messages,
                    tools=self.tool_executor.schemas(),
                    deadline=context.deadline,
                )
                yield self._event(
                    context,
                    AgentEventType.MODEL_COMPLETED,
                    step=step,
                    finish_reason=response.finish_reason,
                    tool_call_count=len(response.get_tool_calls()),
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    attempts=response.attempts,
                    duration_ms=response.duration_ms,
                )

                if not response.get_tool_calls():
                    answer = response.get_content()
                    context.add_message("assistant", answer)
                    context.transition_to(AgentState.COMPLETED)
                    yield self._event(
                        context,
                        AgentEventType.FINAL_ANSWER,
                        success=True,
                        answer=answer,
                        stop_reason="completed",
                        steps=context.step,
                    )
                    return

                self._record_assistant_tool_call(context, response)
                context.transition_to(AgentState.EXECUTING_TOOLS)
                context.pending_tools = list(response.get_tool_calls())
                async for event in self._execute_tools_stream(
                    context,
                    context.pending_tools,
                    record_request=False,
                ):
                    yield event
                context.pending_tools.clear()
                context.transition_to(AgentState.CALLING_MODEL)

            answer = f"Agent 在 {self.max_steps} 步内没有生成最终答案"
            context.transition_to(AgentState.FAILED)
            yield self._event(
                context,
                AgentEventType.RUN_FAILED,
                success=False,
                answer=answer,
                stop_reason="max_steps_exceeded",
                steps=context.step,
                error_code="agent_max_steps",
                error_message=answer,
            )
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
            yield self._event(
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
            yield self._event(
                context,
                AgentEventType.RUN_FAILED,
                success=False,
                stop_reason="error",
                steps=context.step,
                error_code="agent_runtime_error",
                error_message=str(exc),
            )

    async def invoke(self, context: AgentRunContext) -> AgentResult:
        """收集 ``stream`` 的终态，提供非流式调用接口。"""
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

    async def _execute_tools_stream(
        self,
        context: AgentRunContext,
        tool_requests: Sequence[ToolRequest],
        *,
        record_request: bool,
    ) -> AsyncIterator[AgentEvent]:
        if record_request and tool_requests:
            context.add_message(
                "assistant",
                "",
                tool_calls=[request.to_openai() for request in tool_requests],
            )

        for request in tool_requests:
            yield self._event(
                context,
                AgentEventType.TOOL_STARTED,
                name=request.name,
                tool_call_id=request.tool_call_id,
                arguments=request.arguments,
            )

        results = await self.tool_executor.execute_many(
            tool_requests,
            deadline=context.deadline,
        )
        for result in results:
            context.add_message(
                "tool",
                result.content,
                tool_call_id=result.tool_call_id,
            )
            yield self._event(
                context,
                AgentEventType.TOOL_COMPLETED,
                result=result.to_dict(),
            )
        context.last_tool_results = results

    @staticmethod
    def _validate_context(context: AgentRunContext) -> None:
        if not isinstance(context, AgentRunContext):
            raise TypeError("context 必须是 AgentRunContext")
        if context.state is not AgentState.CREATED:
            raise ValueError("AgentRunContext 必须从 created 状态开始")

    @staticmethod
    def _record_assistant_tool_call(
        context: AgentRunContext,
        response: LLMResponse,
    ) -> None:
        metadata = {"tool_calls": response.get_tool_calls_as_dicts()}
        reasoning_content = response.get_reasoning_content()
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content
        context.add_message("assistant", response.get_content(), **metadata)

    @staticmethod
    def _event(
        context: AgentRunContext,
        event_type: AgentEventType,
        **data,
    ) -> AgentEvent:
        payload = {"state": context.state.value}
        payload.update(data)
        return AgentEvent(
            type=event_type,
            run_id=context.run_id,
            session_id=context.session_id,
            data=payload,
        )

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

    @staticmethod
    def _format_direct_results(results: list[ToolResult]) -> str:
        if not results:
            return ""
        if len(results) == 1:
            result = results[0]
            return result.display_content or result.content
        return json.dumps(
            [result.to_dict() for result in results],
            ensure_ascii=False,
            default=str,
        )
