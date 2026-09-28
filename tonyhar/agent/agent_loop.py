"""只负责模型推理与模型发起的 Tool 循环。"""

from collections.abc import AsyncIterator

from tonyhar.tooling import ToolManager

from .events import build_event
from .llm import (
    BaseLLM,
    LLMCompleted,
    LLMResponse,
    LLMTextDelta,
    ModelResponseError,
)
from .runnables import AgentEvent, AgentEventType, AgentRunContext, AgentState
from .tool_runner import ToolRunner


class AgentLoop:
    """运行 LLM → Tool → LLM 循环，不承担意图分类或直接命令路由。"""

    def __init__(
        self,
        llm: BaseLLM,
        tool_manager: ToolManager,
        tool_runner: ToolRunner,
        max_steps: int,
    ) -> None:
        self.llm = llm
        self.tool_manager = tool_manager
        self.tool_runner = tool_runner
        self.max_steps = max_steps

    async def stream(
        self,
        context: AgentRunContext,
        *,
        allowed_tool_names: tuple[str, ...],
    ) -> AsyncIterator[AgentEvent]:
        
        context.transition_to(AgentState.CALLING_MODEL)
        for step in range(1, self.max_steps + 1):
            context.step = step
            yield build_event(
                context,
                AgentEventType.MODEL_STARTED,
                step=step,
            )

            response: LLMResponse | None = None
            async for model_event in self.llm.stream(
                messages=context.messages,
                tools=self.tool_manager.schemas(allowed_tool_names),
                deadline=context.deadline,
            ):
                if isinstance(model_event, LLMTextDelta):
                    yield build_event(
                        context,
                        AgentEventType.TEXT_DELTA,
                        step=step,
                        delta=model_event.text,
                    )
                elif isinstance(model_event, LLMCompleted):
                    response = model_event.response

            if response is None:
                raise ModelResponseError("模型流没有产生完成事件")

            yield build_event(
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
                yield build_event(
                    context,
                    AgentEventType.FINAL_ANSWER,
                    success=True,
                    answer=answer,
                    stop_reason="completed",
                    steps=context.step,
                )
                return

            unauthorized_tools = {
                request.name
                for request in response.get_tool_calls()
                if request.name not in allowed_tool_names
            }
            if unauthorized_tools:
                names = ", ".join(sorted(unauthorized_tools))
                raise ModelResponseError(f"模型请求了本轮未授权工具: {names}")

            self._record_assistant_tool_call(context, response)
            context.transition_to(AgentState.EXECUTING_TOOLS)
            context.pending_tools = list(response.get_tool_calls())
            async for event in self.tool_runner.stream(
                context,
                context.pending_tools,
                record_request=False,
            ):
                yield event
            context.pending_tools.clear()
            context.transition_to(AgentState.CALLING_MODEL)

        answer = f"Agent 在 {self.max_steps} 步内没有生成最终答案"
        context.transition_to(AgentState.FAILED)
        yield build_event(
            context,
            AgentEventType.RUN_FAILED,
            success=False,
            answer=answer,
            stop_reason="max_steps_exceeded",
            steps=context.step,
            error_code="agent_max_steps",
            error_message=answer,
        )

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
