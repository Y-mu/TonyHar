import json
import unittest

from fastapi.testclient import TestClient

from tonyhar.agent.runnables import (
    AgentEvent,
    AgentEventType,
    AgentResult,
    AgentRunContext,
    AgentState,
)
from tonyhar.conversation import ChatService, InMemorySessionStore
from tonyhar.web.app import create_app


class WebTestAgent:
    def __init__(self) -> None:
        self.closed = False

    async def stream(self, context: AgentRunContext):
        yield AgentEvent(
            type=AgentEventType.RUN_STARTED,
            run_id=context.run_id,
            session_id=context.session_id,
            data={"state": context.state.value},
        )
        yield AgentEvent(
            type=AgentEventType.TEXT_DELTA,
            run_id=context.run_id,
            session_id=context.session_id,
            data={"state": context.state.value, "step": 1, "delta": "answer:"},
        )

        context.transition_to(AgentState.DISPATCHING)
        context.add_message("user", context.user_input)
        answer = f"answer:{context.user_input}"
        context.add_message("assistant", answer)
        context.transition_to(AgentState.COMPLETED)
        yield AgentEvent(
            type=AgentEventType.FINAL_ANSWER,
            run_id=context.run_id,
            session_id=context.session_id,
            data={
                "state": context.state.value,
                "success": True,
                "answer": answer,
                "stop_reason": "completed",
                "steps": 0,
            },
        )

    async def invoke(self, context: AgentRunContext) -> AgentResult:
        terminal = None
        async for event in self.stream(context):
            terminal = event
        assert terminal is not None
        return AgentResult(
            run_id=terminal.run_id,
            session_id=terminal.session_id,
            success=True,
            answer=terminal.data["answer"],
        )

    async def aclose(self) -> None:
        self.closed = True


class WebApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = WebTestAgent()
        self.service = ChatService(
            self.agent,
            InMemorySessionStore(),
            system_prompt="system",
        )
        self.app = create_app(lambda: self.service)

    def test_health_and_lifecycle(self):
        with TestClient(self.app) as client:
            self.assertEqual(client.get("/health/live").json(), {"status": "ok"})
            self.assertEqual(
                client.get("/health/ready").json(),
                {"status": "ready"},
            )

        self.assertTrue(self.agent.closed)

    def test_create_stream_and_read_session(self):
        with TestClient(self.app) as client:
            created = client.post("/api/v1/sessions")
            self.assertEqual(created.status_code, 201)
            session_id = created.json()["session_id"]

            response = client.post(
                f"/api/v1/sessions/{session_id}/messages/stream",
                json={"message": " hello "},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(
                response.headers["content-type"].startswith(
                    "text/event-stream"
                )
            )
            self.assertEqual(response.headers["x-accel-buffering"], "no")
            self.assertIn("event: run_started", response.text)
            self.assertIn("event: text_delta", response.text)
            self.assertIn("event: final_answer", response.text)

            data_lines = [
                line.removeprefix("data: ")
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            terminal = json.loads(data_lines[-1])
            self.assertEqual(terminal["type"], "final_answer")
            self.assertEqual(terminal["data"]["answer"], "answer:hello")

            session = client.get(f"/api/v1/sessions/{session_id}")
            self.assertEqual(session.status_code, 200)
            self.assertEqual(
                session.json()["messages"],
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "answer:hello"},
                ],
            )

    def test_request_validation_and_missing_session(self):
        with TestClient(self.app) as client:
            invalid_id = client.post(
                "/api/v1/sessions/not%20valid/messages/stream",
                json={"message": "hello"},
            )
            self.assertEqual(invalid_id.status_code, 422)

            empty_message = client.post(
                "/api/v1/sessions/valid/messages/stream",
                json={"message": "   "},
            )
            self.assertEqual(empty_message.status_code, 422)

            missing = client.get("/api/v1/sessions/missing")
            self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
