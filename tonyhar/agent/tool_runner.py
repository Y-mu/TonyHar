"""Tool 执行与 AgentEvent 适配。"""

from collections.abc import AsyncIterator, Sequence

from tonyhar.tooling import ToolManager, ToolRequest

from .events import build_event
from .runnables import AgentEvent, AgentEventType, AgentRunContext


class ToolRunner:
    """通过 ToolManager 执行请求，并维护本轮消息和事件。"""

    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager

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
        for result in results:
            context.add_message(
                "tool",
                result.content,
                tool_call_id=result.tool_call_id,
            )
            yield build_event(
                context,
                AgentEventType.TOOL_COMPLETED,
                result=result.to_dict(),
            )
        context.last_tool_results = results
