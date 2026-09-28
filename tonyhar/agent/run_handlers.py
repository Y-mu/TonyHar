"""三种稳定运行策略的 Handler。"""

import json
from collections.abc import AsyncIterator
from typing import Protocol

from tonyhar.tooling import ToolResult

from .agent_loop import AgentLoop
from .dispatcher import RunCommand
from .events import build_event
from .runnables import AgentEvent, AgentEventType, AgentRunContext, AgentState
from .tool_runner import ToolRunner


class RunHandler(Protocol):
    async def stream(
        self,
        context: AgentRunContext,
        command: RunCommand,
    ) -> AsyncIterator[AgentEvent]:
        ...


class DirectToolHandler:
    def __init__(self, tool_runner: ToolRunner):
        self.tool_runner = tool_runner

    async def stream(
        self,
        context: AgentRunContext,
        command: RunCommand,
    ) -> AsyncIterator[AgentEvent]:
        context.transition_to(AgentState.EXECUTING_TOOLS)
        context.pending_tools = list(command.tool_requests)
        async for event in self.tool_runner.stream(
            context,
            command.tool_requests,
            record_request=True,
        ):
            yield event
        context.pending_tools.clear()

        results = list(context.last_tool_results)
        answer = self._format_results(results)
        context.add_message("assistant", answer)
        failed_result = next(
            (result for result in results if not result.success),
            None,
        )
        if failed_result is not None:
            context.transition_to(AgentState.FAILED)
            yield build_event(
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
        yield build_event(
            context,
            AgentEventType.FINAL_ANSWER,
            success=True,
            answer=answer,
            stop_reason="completed",
            steps=context.step,
        )

    @staticmethod
    def _format_results(results: list[ToolResult]) -> str:
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


class RetrievalAgentHandler:
    def __init__(self, tool_runner: ToolRunner, agent_loop: AgentLoop):
        self.tool_runner = tool_runner
        self.agent_loop = agent_loop

    async def stream(
        self,
        context: AgentRunContext,
        command: RunCommand,
    ) -> AsyncIterator[AgentEvent]:
        context.transition_to(AgentState.EXECUTING_TOOLS)
        context.pending_tools = list(command.tool_requests)
        async for event in self.tool_runner.stream(
            context,
            command.tool_requests,
            record_request=True,
        ):
            yield event
        context.pending_tools.clear()

        async for event in self.agent_loop.stream(
            context,
            allowed_tool_names=command.model_tool_names,
        ):
            yield event


class AgentHandler:
    def __init__(self, agent_loop: AgentLoop):
        self.agent_loop = agent_loop

    async def stream(
        self,
        context: AgentRunContext,
        command: RunCommand,
    ) -> AsyncIterator[AgentEvent]:
        async for event in self.agent_loop.stream(
            context,
            allowed_tool_names=command.model_tool_names,
        ):
            yield event
