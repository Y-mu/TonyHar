import asyncio
from typing import Dict, Iterable, List

from LLM.LLMResponse import ToolCall
from .tool import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool], timeout_seconds: float = 30.0) -> None:
        self._tools: Dict[str, Tool] = {}
        self._timeout_seconds = timeout_seconds
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self) -> List[dict]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}} for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(False, error=f"Unknown tool: {call.name}")
        try:
            return await asyncio.wait_for(tool.execute(call.arguments), self._timeout_seconds)
        except asyncio.TimeoutError:
            return ToolResult(False, error=f"Tool timed out: {call.name}")
        except Exception as error:
            return ToolResult(False, error=f"Tool failed: {error}")
