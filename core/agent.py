"""Agent 运行时编排器。"""

import json

from tooling import ToolExecutor, ToolRegistry, ToolRequest, ToolResult

from .intent_planner import ExecutionMode, IntentPlanner
from .llm import BaseLLM, LLMResponse
from .memory import Memory


class Agent:
    """编排路由计划、模型调用、工具结果和会话消息。"""

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        intent_planner: IntentPlanner,
        system_prompt: str = "",
        memory: Memory | None = None,
        max_steps: int = 10,
    ):
        self.llm = llm
        self.tools = tools
        self.intent_planner = intent_planner
        self.memory = memory or Memory()
        self.max_steps = max_steps
        if system_prompt:
            self.memory.add("system", system_prompt)

    async def invoke(self, user_input: str) -> str:
        plan = self.intent_planner.plan(user_input)
        self.memory.add("user", plan.user_message)

        planned_results = self._execute_planned_tools(plan.tool_calls)
        if plan.mode is ExecutionMode.DIRECT_TOOL:
            answer = self._format_direct_results(planned_results)
            self.memory.add("assistant", answer)
            return answer

        for _step in range(self.max_steps):
            response = await self.llm.chat(
                messages=self.memory.get(),
                tools=self.tools.schemas(),
            )

            if not response.get_tool_calls():
                answer = response.get_content()
                self.memory.add("assistant", answer)
                return answer

            self._record_assistant_tool_call(response)
            results = ToolExecutor.invoke_many(
                response.get_tool_calls(),
                self.tools,
            )
            self._record_tool_results(results)

        return f"Agent 在 {self.max_steps} 步内没有生成最终答案"

    def _execute_planned_tools(
        self,
        tool_requests: tuple[ToolRequest, ...],
    ) -> list[ToolResult]:
        if not tool_requests:
            return []

        self.memory.add(
            "assistant",
            "",
            tool_calls=[request.to_openai() for request in tool_requests],
        )
        results = ToolExecutor.invoke_many(tool_requests, self.tools)
        self._record_tool_results(results)
        return results

    def _record_assistant_tool_call(self, response: LLMResponse) -> None:
        metadata = {"tool_calls": response.get_tool_calls_as_dicts()}
        reasoning_content = response.get_reasoning_content()
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content
        self.memory.add("assistant", response.get_content(), **metadata)

    def _record_tool_results(self, results: list[ToolResult]) -> None:
        for result in results:
            self.memory.add(
                "tool",
                result.content,
                tool_call_id=result.tool_call_id,
            )

    @staticmethod
    def _format_direct_results(results: list[ToolResult]) -> str:
        if not results:
            return ""
        if len(results) == 1:
            return results[0].content
        return json.dumps(
            [result.to_dict() for result in results],
            ensure_ascii=False,
            default=str,
        )
