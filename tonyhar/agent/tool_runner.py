"""Tool 执行与 AgentEvent 适配。"""

from collections.abc import AsyncIterator, Sequence

from tonyhar.tooling import ToolManager, ToolRequest

from .events import build_event
from .runnables import AgentEvent, AgentEventType, AgentRunContext


class ToolRunner:
    """通过 ToolManager 执行请求，并维护本轮消息和事件。"""

    def __init__(
        self,
        tool_manager: ToolManager,
        *,
        max_total_model_output_chars: int = 12_000,
    ) -> None:
        if max_total_model_output_chars < 256:
            raise ValueError("max_total_model_output_chars 不能小于 256")
        self.tool_manager = tool_manager
        self.max_total_model_output_chars = max_total_model_output_chars

    async def stream(
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
            yield build_event(
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
        remaining_chars = max(
            0,
            self.max_total_model_output_chars - context.tool_context_chars_used,
        )
        for result in results:
            tool = self.tool_manager.get(result.name)
            per_tool_limit = (
                tool.policy.max_model_output_chars
                if tool is not None
                else remaining_chars
            )
            if remaining_chars < 256:
                model_content = (
                    '{"truncated":true,"reason":"total_tool_output_limit"}'
                )
            else:
                limit = min(per_tool_limit, remaining_chars)
                model_content = result.content_for_model(limit)
            remaining_chars = max(0, remaining_chars - len(model_content))
            context.tool_context_chars_used += len(model_content)
            context.add_message(
                "tool",
                model_content,
                tool_call_id=result.tool_call_id,
            )
            yield build_event(
                context,
                AgentEventType.TOOL_COMPLETED,
                result=result.to_dict(),
            )
        context.last_tool_results = results
