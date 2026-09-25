import unittest
import asyncio

from tonyhar.agent import Agent
from tonyhar.agent.intent_planner import IntentPlanner
from tonyhar.agent.llm import BaseLLM, LLMResponse
from tonyhar.agent.runnables import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    AgentRunContext,
    AgentState,
)
from tonyhar.agent.user_intent_recognizer import IntentResult
from tonyhar.resilience import Deadline
from tonyhar.tooling import ToolRegistry, ToolRequest, tool


class ChatRecognizer:
    def classify(self, text):
        return IntentResult(name="chat", score=0.9)


class EventLLM(BaseLLM):
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, deadline):
        self.calls += 1
        return LLMResponse(content="完成")

    async def aclose(self):
        return None


class AgentRunnablesTest(unittest.TestCase):
    def test_context_copies_messages_and_validates_transitions(self):
        source_messages = [{"role": "system", "content": "test"}]
        context = AgentRunContext(
            run_id="run-1",
            session_id="session-1",
            user_input="你好",
            messages=source_messages,
            deadline=Deadline.after(1),
        )

        context.messages[0]["content"] = "changed"
        self.assertEqual(source_messages[0]["content"], "test")

        context.transition_to(AgentState.PLANNING)
        context.transition_to(AgentState.CALLING_MODEL)
        context.transition_to(AgentState.COMPLETED)
        self.assertIs(context.state, AgentState.COMPLETED)

        with self.assertRaisesRegex(ValueError, "非法 Agent 状态转换"):
            context.transition_to(AgentState.PLANNING)

    def test_event_and_result_are_structured_protocols(self):
        event = AgentEvent(
            type=AgentEventType.RUN_STARTED,
            run_id="run-1",
            session_id="session-1",
        )
        result = AgentResult(
            run_id="run-1",
            session_id="session-1",
            success=True,
            answer="完成",
        )

        self.assertIs(event.type, AgentEventType.RUN_STARTED)
        self.assertTrue(result.success)
        self.assertEqual(result.answer, "完成")

    def test_agent_stream_emits_terminal_event_and_hides_reasoning(self):
        async def run():
            agent = Agent(
                llm=EventLLM(),
                tools=ToolRegistry(),
                intent_planner=IntentPlanner(ChatRecognizer()),
            )
            context = AgentRunContext(
                run_id="run-stream",
                session_id="session-stream",
                user_input="你好",
                messages=[],
                deadline=Deadline.after(1),
            )
            events = [event async for event in agent.stream(context)]
            return events, context

        events, context = asyncio.run(run())
        self.assertEqual(
            [event.type for event in events],
            [
                AgentEventType.RUN_STARTED,
                AgentEventType.INTENT_PLANNED,
                AgentEventType.MODEL_STARTED,
                AgentEventType.MODEL_COMPLETED,
                AgentEventType.FINAL_ANSWER,
            ],
        )
        self.assertEqual(events[-1].data["answer"], "完成")
        self.assertEqual(context.state, AgentState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
