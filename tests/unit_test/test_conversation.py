import asyncio
import unittest

from tonyhar.conversation import ChatService, InMemorySessionStore, Session
from tonyhar.agent.runnables import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    AgentRunContext,
    AgentState,
)


class RecordingAgent:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.contexts: list[AgentRunContext] = []
        self.active_by_session: dict[str, int] = {}
        self.max_active_by_session: dict[str, int] = {}
        self.active_total = 0
        self.max_active_total = 0
        self.closed = False

    async def invoke(self, context: AgentRunContext) -> AgentResult:
        self.contexts.append(context)
        session_id = context.session_id
        self.active_by_session[session_id] = (
            self.active_by_session.get(session_id, 0) + 1
        )
        self.max_active_by_session[session_id] = max(
            self.max_active_by_session.get(session_id, 0),
            self.active_by_session[session_id],
        )
        self.active_total += 1
        self.max_active_total = max(self.max_active_total, self.active_total)

        try:
            context.transition_to(AgentState.PLANNING)
            context.add_message("user", context.user_input)
            if self.delay:
                await asyncio.sleep(self.delay)
            answer = f"answer:{context.user_input}"
            context.add_message("assistant", answer)
            context.transition_to(AgentState.COMPLETED)
            return AgentResult(
                run_id=context.run_id,
                session_id=context.session_id,
                success=True,
                answer=answer,
            )
        finally:
            self.active_by_session[session_id] -= 1
            self.active_total -= 1

    async def stream(self, context: AgentRunContext):
        result = await self.invoke(context)
        yield AgentEvent(
            type=(
                AgentEventType.FINAL_ANSWER
                if result.success
                else AgentEventType.RUN_FAILED
            ),
            run_id=result.run_id,
            session_id=result.session_id,
            data={
                "success": result.success,
                "answer": result.answer,
                "stop_reason": result.stop_reason,
                "steps": result.steps,
            },
        )

    async def aclose(self):
        self.closed = True


class ConversationTest(unittest.IsolatedAsyncioTestCase):
    async def test_sessions_keep_independent_history(self):
        agent = RecordingAgent()
        store = InMemorySessionStore()
        service = ChatService(agent, store, system_prompt="system")

        await service.invoke("session-a", "A1")
        await service.invoke("session-b", "B1")
        await service.invoke("session-a", "A2")

        session_a = await store.get("session-a")
        session_b = await store.get("session-b")
        self.assertIsNotNone(session_a)
        self.assertIsNotNone(session_b)
        assert session_a is not None
        assert session_b is not None

        a_contents = [message["content"] for message in session_a.messages]
        b_contents = [message["content"] for message in session_b.messages]
        self.assertEqual(session_a.messages[0]["role"], "system")
        self.assertEqual(session_b.messages[0]["role"], "system")
        self.assertIn("A1", a_contents)
        self.assertIn("A2", a_contents)
        self.assertNotIn("B1", a_contents)
        self.assertIn("B1", b_contents)
        self.assertNotIn("A1", b_contents)

    async def test_same_session_is_serial_and_different_sessions_overlap(self):
        agent = RecordingAgent(delay=0.02)
        service = ChatService(agent, InMemorySessionStore())

        await asyncio.gather(
            service.invoke("same", "first"),
            service.invoke("same", "second"),
        )
        self.assertEqual(agent.max_active_by_session["same"], 1)

        await asyncio.gather(
            service.invoke("one", "first"),
            service.invoke("two", "second"),
        )
        self.assertGreaterEqual(agent.max_active_total, 2)

    async def test_message_trimming_keeps_tool_exchange_together(self):
        session = Session.create("session", max_turns=1)
        session.replace_messages([
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "old-call"}],
            },
            {
                "role": "tool",
                "content": "old-result",
                "tool_call_id": "old-call",
            },
            {"role": "assistant", "content": "old-answer"},
            {"role": "user", "content": "new"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "new-call"}],
            },
            {
                "role": "tool",
                "content": "new-result",
                "tool_call_id": "new-call",
            },
            {"role": "assistant", "content": "new-answer"},
        ])

        contents = [message["content"] for message in session.messages]
        self.assertEqual(contents, ["new", "", "new-result", "new-answer"])

    async def test_chat_service_stream_saves_context_after_event_stream(self):
        service = ChatService(RecordingAgent(), InMemorySessionStore())

        events = [
            event async for event in service.stream("stream-session", "hello")
        ]

        self.assertEqual(events[-1].type, AgentEventType.FINAL_ANSWER)
        session = await service._sessions.get("stream-session")
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(
            [message["content"] for message in session.messages],
            ["hello", "answer:hello"],
        )

    async def test_get_session_returns_an_independent_snapshot(self):
        service = ChatService(RecordingAgent(), InMemorySessionStore())
        await service.invoke("session", "hello")

        snapshot = await service.get_session("session")
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        snapshot.messages.clear()

        stored = await service.get_session("session")
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(len(stored.messages), 2)

    async def test_stream_cancellation_saves_state_and_releases_lock(self):
        agent = RecordingAgent(delay=10.0)
        store = InMemorySessionStore()
        service = ChatService(agent, store)

        async def consume() -> None:
            async for _ in service.stream("cancelled", "partial"):
                pass

        task = asyncio.create_task(consume())
        for _ in range(100):
            if agent.contexts:
                break
            await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        saved = await store.get("cancelled")
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(
            [message["content"] for message in saved.messages],
            ["partial"],
        )

        agent.delay = 0
        result = await asyncio.wait_for(
            service.invoke("cancelled", "next"),
            timeout=0.5,
        )
        self.assertTrue(result.success)

    async def test_close_releases_agent_resources(self):
        agent = RecordingAgent()
        service = ChatService(agent, InMemorySessionStore())

        await service.aclose()

        self.assertTrue(agent.closed)


if __name__ == "__main__":
    unittest.main()
