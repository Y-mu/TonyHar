"""工具实例装配和异步运行时管理。"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import Callable, Sequence

from tonyhar.resilience import Deadline, RunDeadlineExceeded

from .base import BaseTool, RetryableToolError, ToolDefinition
from .models import ToolRequest, ToolResult

ToolFactory = Callable[[], BaseTool]
ToolProvider = BaseTool | type[BaseTool] | ToolFactory


class ToolManager:
    """构建、注册并执行全部业务工具。

    ``ToolManager`` 是 Agent 访问工具的唯一边界。它在初始化时从工具实例、
    ``BaseTool`` 类型或 factory 构建工具实例，之后统一提供缓存 Schema、
    超时、重试、并发和批量执行能力。工具实例不会在每次请求中重复创建。
    """

    def __init__(
        self,
        providers: Sequence[ToolProvider] = (),
        *,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须大于 0")
        self._tools: dict[str, BaseTool] = {}
        self._definitions: dict[str, ToolDefinition] = {}
        self._schemas: tuple[dict, ...] = ()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        for provider in providers:
            self.register(provider)

    def register(self, provider: ToolProvider) -> BaseTool:
        """构建并注册一个工具实例。"""
        if isinstance(provider, BaseTool):
            tool = provider
        elif inspect.isclass(provider) and issubclass(provider, BaseTool):
            tool = provider()
        elif callable(provider):
            tool = provider()
        else:
            raise TypeError("工具 provider 必须是 BaseTool、BaseTool 类型或 factory")
        if not isinstance(tool, BaseTool):
            raise TypeError("工具 provider 必须返回 BaseTool")
        if tool.name in self._tools:
            raise ValueError(f"工具名称重复: {tool.name}")
        self._tools[tool.name] = tool
        self._definitions[tool.definition.name] = tool.definition
        self._schemas = tuple(
            tool.to_schema() for tool in self._tools.values()
        )
        return tool

    def register_many(self, *providers: ToolProvider) -> None:
        """按传入顺序构建并注册多个工具实例。"""
        for provider in providers:
            self.register(provider)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        """返回缓存的 OpenAI Function Calling Schema。"""
        return list(self._schemas)

    async def execute(
        self,
        tool_request: ToolRequest,
        *,
        deadline: Deadline,
    ) -> ToolResult:
        if not isinstance(tool_request, ToolRequest):
            raise TypeError("tool_request 必须是 ToolRequest")

        tool = self._tools.get(tool_request.name)
        if tool is None:
            return ToolResult(
                name=tool_request.name,
                success=False,
                tool_call_id=tool_request.tool_call_id,
                error_code="tool_not_found",
                error_message=f"工具 '{tool_request.name}' 不存在",
            )

        started_at = time.monotonic()
        policy = tool.policy
        for attempt in range(1, policy.max_attempts + 1):
            remaining = deadline.require_remaining()
            timeout = min(policy.timeout_seconds, remaining)
            bounded_by_run = remaining <= policy.timeout_seconds
            try:
                async with self._semaphore:
                    async with asyncio.timeout(timeout):
                        data = await tool.execute(**tool_request.arguments)
                return ToolResult(
                    name=tool_request.name,
                    success=True,
                    tool_call_id=tool_request.tool_call_id,
                    data=data,
                    display_content=tool.format_result(data),
                    attempts=attempt,
                    duration_ms=self._duration_ms(started_at),
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                if bounded_by_run and deadline.remaining() <= 0:
                    raise RunDeadlineExceeded()
                error_code = "tool_timeout"
                error_message = (
                    f"工具 '{tool_request.name}' 单次执行超过 "
                    f"{policy.timeout_seconds:g} 秒"
                )
                retryable = policy.idempotent
            except RetryableToolError as exc:
                error_code = "tool_unavailable"
                error_message = str(exc)
                retryable = policy.idempotent
            except Exception as exc:
                error_code = "tool_execution_error"
                error_message = str(exc)
                retryable = False

            if retryable and attempt < policy.max_attempts:
                ceiling = min(
                    policy.retry_max_delay,
                    policy.retry_base_delay * (2 ** (attempt - 1)),
                )
                await deadline.sleep(random.uniform(0.0, ceiling))
                continue

            return ToolResult(
                name=tool_request.name,
                success=False,
                tool_call_id=tool_request.tool_call_id,
                error_code=error_code,
                error_message=error_message,
                attempts=attempt,
                duration_ms=self._duration_ms(started_at),
            )

        raise AssertionError("工具重试循环未返回结果")

    async def execute_many(
        self,
        tool_requests: Sequence[ToolRequest],
        *,
        deadline: Deadline,
    ) -> list[ToolResult]:
        """并发执行安全工具；副作用工具与前后批次保持串行。"""
        results: list[ToolResult | None] = [None] * len(tool_requests)
        pending: list[tuple[int, ToolRequest]] = []

        async def flush_parallel() -> None:
            if not pending:
                return
            batch = list(pending)
            pending.clear()
            batch_results = await asyncio.gather(*(
                self.execute(request, deadline=deadline)
                for _, request in batch
            ))
            for (index, _), result in zip(batch, batch_results, strict=True):
                results[index] = result

        for index, request in enumerate(tool_requests):
            tool = self._tools.get(request.name)
            if tool is not None and tool.policy.parallel_safe:
                pending.append((index, request))
                continue

            await flush_parallel()
            results[index] = await self.execute(request, deadline=deadline)

        await flush_parallel()
        if any(result is None for result in results):
            raise AssertionError("工具批次存在缺失结果")
        return [result for result in results if result is not None]

    @staticmethod
    def _duration_ms(started_at: float) -> float:
        return round((time.monotonic() - started_at) * 1000, 3)
