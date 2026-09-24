"""统一工具执行入口。"""

from collections.abc import Sequence

from .models import ToolRequest, ToolResult
from .registry import ToolRegistry


class ToolExecutor:
    """无状态调用器：只接受 ToolRequest，只返回 ToolResult。"""

    @staticmethod
    def invoke(
        tool_request: ToolRequest,
        tool_registry: ToolRegistry,
    ) -> ToolResult:
        if not isinstance(tool_request, ToolRequest):
            raise TypeError("tool_request 必须是 ToolRequest")
        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry 必须是 ToolRegistry")

        tool = tool_registry.get(tool_request.name)
        if tool is None:
            return ToolResult(
                name=tool_request.name,
                success=False,
                tool_call_id=tool_request.tool_call_id,
                error_code="tool_not_found",
                error_message=f"工具 '{tool_request.name}' 不存在",
            )

        try:
            data = tool.run(**tool_request.arguments)
            return ToolResult(
                name=tool_request.name,
                success=True,
                tool_call_id=tool_request.tool_call_id,
                data=data,
            )
        except Exception as exc:
            return ToolResult(
                name=tool_request.name,
                success=False,
                tool_call_id=tool_request.tool_call_id,
                error_code="tool_execution_error",
                error_message=str(exc),
            )

    @staticmethod
    def invoke_many(
        tool_requests: Sequence[ToolRequest],
        tool_registry: ToolRegistry,
    ) -> list[ToolResult]:
        return [
            ToolExecutor.invoke(request, tool_registry)
            for request in tool_requests
        ]
