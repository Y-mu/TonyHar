"""Agent 状态机运行时。"""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

from tonyhar.resilience import ExecutionError, RunDeadlineExceeded
from tonyhar.tooling import ToolManager, ToolRequest, ToolResult

from .intent_planner import ExecutionMode, IntentPlanner
from .llm import (
    BaseLLM,
    LLMCompleted,
    LLMResponse,
    LLMTextDelta,
    ModelResponseError,
)
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

    ``Agent`` 只保存可复用的基础设施依赖：LLM、工具管理器、意图规划器
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
        tool_manager: ToolManager,
        intent_planner: IntentPlanner,
        max_steps: int = 10,
    ):
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        self.llm = llm
        if not isinstance(tool_manager, ToolManager):
            raise TypeError("tool_manager 必须是 ToolManager")
        self.tool_manager = tool_manager
        self.intent_planner = intent_planner
        self.max_steps = max_steps

    async def stream(
        self,
        context: AgentRunContext,
    ) -> AsyncIterator[AgentEvent]:
        """执行一次独立运行，并按状态机推进顺序产生事件。"""
        # 校验本轮独立 context 必须从 CREATED 开始，并发出运行开始事件。
        self._validate_context(context)
        yield self._event(
            context,
            AgentEventType.RUN_STARTED,
            state=context.state.value,
        )

        try:
            # 进入 PLANNING，由 IntentPlanner 生成本轮执行计划。
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

            # 优先执行计划中已经确定的工具调用，并将结果写回 context。
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

            # DIRECT_TOOL 模式直接格式化工具结果，不再调用模型。
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

            # 进入模型推理循环，直到生成最终文本或达到 max_steps。
            context.transition_to(AgentState.CALLING_MODEL)
            for step in range(1, self.max_steps + 1):
                context.step = step
                yield self._event(
                    context,
                    AgentEventType.MODEL_STARTED,
                    step=step,
                )
                response: LLMResponse | None = None
                async for model_event in self.llm.stream(
                    messages=context.messages,
                    tools=self.tool_manager.schemas(),
                    deadline=context.deadline,
                ):
                    if isinstance(model_event, LLMTextDelta):
                        yield self._event(
                            context,
                            AgentEventType.TEXT_DELTA,
                            step=step,
                            delta=model_event.text,
                        )
                    elif isinstance(model_event, LLMCompleted):
                        response = model_event.response

                if response is None:
                    raise ModelResponseError("模型流没有产生完成事件")

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

                # 模型没有请求工具时，将普通文本作为最终答案完成本轮运行。
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

                # 记录模型的工具调用，执行完成后回到 CALLING_MODEL 继续推理。
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

            # 达到最大步数仍未生成最终答案时，以失败终态结束。
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
            # 取消必须进入 CANCELLED，并继续向上抛出取消异常。
            if context.state not in {
                AgentState.COMPLETED,
                AgentState.FAILED,
                AgentState.CANCELLED,
            }:
                context.transition_to(AgentState.CANCELLED)
            raise
        except ExecutionError as exc:
            # 已知执行异常保留具体错误码，并区分超时与依赖错误。
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
            # 其余未处理异常统一转换为 FAILED 事件，不伪装成成功答案。
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

        results = await self.tool_manager.execute_many(
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
