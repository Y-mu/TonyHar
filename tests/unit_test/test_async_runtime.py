import asyncio
import unittest
from types import SimpleNamespace

from tonyhar.agent.llm import (
    DeepSeekLLM,
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


class FakeCompletions:
    def __init__(self, failures=0, delay=0.0):
        self.failures = failures
        self.delay = delay
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.calls <= self.failures:
            raise TimeoutError("upstream timeout")
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )


class FakeClient:
    def __init__(self, failures=0, delay=0.0):
        self.completions = FakeCompletions(failures, delay)
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

        result = await llm.chat([], [], deadline=Deadline.after(1))

        self.assertEqual(result.content, "ok")
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
                await llm.chat([], [], deadline=Deadline.after(1))
        self.assertEqual(llm.circuit_breaker.state, "open")
        with self.assertRaises(ModelCircuitOpenError):
            await llm.chat([], [], deadline=Deadline.after(1))
        self.assertEqual(client.completions.calls, 2)

    async def test_run_deadline_bounds_model_attempt(self):
        llm = DeepSeekLLM(
            api_key="test",
            client=FakeClient(delay=1),
            attempt_timeout=10,
            retry_policy=RetryPolicy(max_attempts=3, base_delay=0),
        )

        with self.assertRaises(RunDeadlineExceeded):
            await llm.chat([], [], deadline=Deadline.after(0.01))

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
