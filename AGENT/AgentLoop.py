import json
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

from LLM.LLMResponse import LLMResponse, ToolCall
from TOOLS.registry import ToolRegistry
from TOOLS.tool import ToolResult

from ..LLM.LMM_Client import LLMClient

Message = Dict[str, Any]



@dataclass(frozen=True)
class AgentRunResult:
    content: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    messages: List[Message]


class AgentMaxStepsError(RuntimeError):
    pass


class AgentLoop:
    def __init__(self, llm: LLMClient, tool_registry: ToolRegistry, max_steps: int = 8) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        self._llm = llm
        self._tool_registry = tool_registry
        self._max_steps = max_steps

    async def run(self, messages: Sequence[Message]) -> AgentRunResult:
        conversation = [dict(message) for message in messages]
        tools = self._tool_registry.schemas()
        prompt_tokens = completion_tokens = 0
        for step in range(1, self._max_steps + 1):
            response = await self._llm.chat(conversation, tools)
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            if not response.tool_calls:
                conversation.append({"role": "assistant", "content": response.content})
                return AgentRunResult(response.content, step, prompt_tokens, completion_tokens, conversation)
            conversation.append(self._assistant_tool_message(response))
            for call in response.tool_calls:
                result = await self._tool_registry.execute(call)
                conversation.append(self._tool_result_message(call, result))
        raise AgentMaxStepsError(f"Agent exceeded maximum steps: {self._max_steps}")

    @staticmethod
    def _assistant_tool_message(response: LLMResponse) -> Message:
        return {"role": "assistant", "content": response.content or None, "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}} for c in response.tool_calls]}

    @staticmethod
    def _tool_result_message(call: ToolCall, result: ToolResult) -> Message:
        payload = {"success": result.success, "output": result.output, "error": result.error, "metadata": result.metadata}
        return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": json.dumps(payload, ensure_ascii=False, default=str)}
