import asyncio
import unittest
from types import SimpleNamespace

from tonyhar.agent.llm import (
    DeepSeekLLM,
    LLMCompleted,
    LLMTextDelta,
    ModelCircuitOpenError,
    ModelTimeoutError,
)
from tonyhar.resilience import Deadline, RetryPolicy, RunDeadlineExceeded
from tonyhar.resilience import CircuitBreaker, CircuitOpenError
from tonyhar.tooling import (
    BaseTool,
    RetryableToolError,
    ToolManager,
    ToolPolicy,
    ToolRequest,
)


class AsyncEchoTool(BaseTool):
    name = "async_echo"
    description = "异步返回输入"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    policy = ToolPolicy(parallel_safe=True)

    async def execute(self, text: str) -> str:
        await asyncio.sleep(0)
        return text


def text_chunk(
    content: str = "",
    *,
    finish_reason: str | None = None,
    tool_calls=None,
):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(
                content=content,
                reasoning_content=None,
                tool_calls=tool_calls,
            ),
            finish_reason=finish_reason,
        )],
        usage=None,
    )


class FakeStream:
    def __init__(self, chunks, delay=0.0, error=None):
        self.chunks = chunks
        self.delay = delay
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk
        if self.error is not None:
            raise self.error


class FakeCompletions:
    def __init__(self, failures=0, delay=0.0, chunks=None, stream_error=None):
        self.failures = failures
        self.delay = delay
        self.chunks = chunks or [text_chunk("ok", finish_reason="stop")]
        self.stream_error = stream_error
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("upstream timeout")
        return FakeStream(
            self.chunks,
            delay=self.delay,
            error=self.stream_error,
        )


class FakeClient:
    def __init__(
        self,
        failures=0,
        delay=0.0,
        chunks=None,
        stream_error=None,
    ):
        self.completions = FakeCompletions(
            failures,
            delay,
            chunks,
            stream_error,
        )
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self):
        self.closed = True


class AsyncToolRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_safe_tools_overlap_and_preserve_order(self):
        active = 0
        max_active = 0

        class TrackedEchoTool(AsyncEchoTool):
            name = "tracked_echo"

            async def execute(self, text: str) -> str:
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                try:
                    await asyncio.sleep(0.01)
                    return text
                finally:
                    active -= 1

        manager = ToolManager([TrackedEchoTool], max_concurrency=2)
        results = await manager.execute_many([
            ToolRequest(name="tracked_echo", arguments={"text": "a"}),
            ToolRequest(name="tracked_echo", arguments={"text": "b"}),
        ], deadline=Deadline.after(1))

        self.assertEqual([result.data for result in results], ["a", "b"])
        self.assertEqual(max_active, 2)

    async def test_idempotent_tool_retries_temporary_failure(self):
        calls = 0

        class FlakyTool(BaseTool):
            name = "flaky_tool"
            description = "首次执行暂时失败"
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
            policy = ToolPolicy(
                max_attempts=2,
                idempotent=True,
                retry_base_delay=0,
                retry_max_delay=0,
            )

            async def execute(self) -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RetryableToolError("temporary")
                return "ok"

        result = await ToolManager([FlakyTool]).execute(
            ToolRequest(name="flaky_tool"),
            deadline=Deadline.after(1),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)

    async def test_tool_timeout_returns_structured_failure(self):
        class SlowTool(BaseTool):
            name = "slow_tool"
            description = "故意超时"
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
            policy = ToolPolicy(timeout_seconds=0.01)

            async def execute(self) -> str:
                await asyncio.sleep(1)
                return "late"

        result = await ToolManager([SlowTool]).execute(
            ToolRequest(name="slow_tool"),
            deadline=Deadline.after(1),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_timeout")


class LLMResilienceTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def collect(llm, deadline=None):
        return [
            event
            async for event in llm.stream(
                [],
                [],
                deadline=deadline or Deadline.after(1),
            )
        ]

    async def test_retry_then_success(self):
        client = FakeClient(failures=2)
        llm = DeepSeekLLM(
            api_key="test",
            client=client,
            retry_policy=RetryPolicy(
                max_attempts=3,
                base_delay=0,
                max_delay=0,
            ),
        )

        events = await self.collect(llm)
        result = next(
            event.response
            for event in events
            if isinstance(event, LLMCompleted)
        )

        self.assertEqual(result.content, "ok")
        self.assertEqual(
            [event.text for event in events if isinstance(event, LLMTextDelta)],
            ["ok"],
        )
        self.assertEqual(client.completions.calls, 3)
        self.assertEqual(llm.circuit_breaker.state, "closed")

    async def test_circuit_opens_after_logical_failures(self):
        client = FakeClient(failures=99)
        llm = DeepSeekLLM(
            api_key="test",
            client=client,
            retry_policy=RetryPolicy(
                max_attempts=1,
                base_delay=0,
                max_delay=0,
            ),
            circuit_failure_threshold=2,
            circuit_recovery_timeout=60,
        )

        for _ in range(2):
            with self.assertRaises(ModelTimeoutError):
                await self.collect(llm)
        self.assertEqual(llm.circuit_breaker.state, "open")
        with self.assertRaises(ModelCircuitOpenError):
            await self.collect(llm)
        self.assertEqual(client.completions.calls, 2)

    async def test_run_deadline_bounds_model_attempt(self):
        llm = DeepSeekLLM(
            api_key="test",
            client=FakeClient(delay=1),
            attempt_timeout=10,
            retry_policy=RetryPolicy(max_attempts=3, base_delay=0),
        )

        with self.assertRaises(RunDeadlineExceeded):
            await self.collect(llm, Deadline.after(0.01))

    async def test_stream_reassembles_fragmented_tool_call(self):
        first_call = SimpleNamespace(
            index=0,
            id="call-",
            function=SimpleNamespace(
                name="knowledge_",
                arguments='{"query":"',
            ),
        )
        second_call = SimpleNamespace(
            index=0,
            id="1",
            function=SimpleNamespace(
                name="search",
                arguments='test"}',
            ),
        )
        client = FakeClient(chunks=[
            text_chunk(tool_calls=[first_call]),
            text_chunk(tool_calls=[second_call], finish_reason="tool_calls"),
        ])
        llm = DeepSeekLLM(api_key="test", client=client)

        events = await self.collect(llm)
        result = next(
            event.response
            for event in events
            if isinstance(event, LLMCompleted)
        )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].tool_call_id, "call-1")
        self.assertEqual(result.tool_calls[0].name, "knowledge_search")
        self.assertEqual(result.tool_calls[0].arguments, {"query": "test"})

    async def test_stream_does_not_retry_after_text_was_emitted(self):
        client = FakeClient(
            chunks=[text_chunk("partial")],
            stream_error=TimeoutError("stream interrupted"),
        )
        llm = DeepSeekLLM(
            api_key="test",
            client=client,
            retry_policy=RetryPolicy(
                max_attempts=3,
                base_delay=0,
                max_delay=0,
            ),
        )

        received = []
        with self.assertRaises(ModelTimeoutError):
            async for event in llm.stream(
                [],
                [],
                deadline=Deadline.after(1),
            ):
                received.append(event)

        self.assertEqual(client.completions.calls, 1)
        self.assertEqual(received, [LLMTextDelta(text="partial")])

    async def test_close_releases_client(self):
        client = FakeClient()
        llm = DeepSeekLLM(api_key="test", client=client)

        await llm.aclose()

        self.assertTrue(client.closed)


class CircuitBreakerConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_half_open_probe_releases_slot(self):
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.001,
        )
        permit = breaker.acquire()
        breaker.record_failure(permit)
        await asyncio.sleep(0.002)

        probe = breaker.acquire()
        with self.assertRaises(CircuitOpenError):
            breaker.acquire()
        breaker.release_cancelled(probe)

        next_probe = breaker.acquire()
        breaker.record_success(next_probe)
        self.assertEqual(breaker.state, "closed")

    async def test_stale_success_cannot_close_new_open_generation(self):
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=60,
        )
        failing = breaker.acquire()
        stale_success = breaker.acquire()

        breaker.record_failure(failing)
        breaker.record_success(stale_success)

        self.assertEqual(breaker.state, "open")


if __name__ == "__main__":
    unittest.main()
