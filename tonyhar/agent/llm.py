"""模型端口与 DeepSeek 异步适配器。"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from tonyhar.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    Deadline,
    ExecutionError,
    RetryPolicy,
    RunDeadlineExceeded,
)
from tonyhar.tooling import ToolRequest


@dataclass(frozen=True)
class LLMResponse:
    content: str = ""
    # DeepSeek thinking mode requires this value to be sent back unchanged
    # when an assistant message contains tool calls.
    reasoning_content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: list[ToolRequest] = field(default_factory=list)
    attempts: int = 1
    duration_ms: float = 0.0

    def get_tool_calls(self) -> list[ToolRequest]:
        return self.tool_calls

    def get_tool_calls_as_dicts(self) -> list[dict]:
        return [call.to_openai() for call in self.tool_calls]

    def get_content(self) -> str:
        return self.content or ""

    def get_reasoning_content(self) -> str:
        return self.reasoning_content or ""


@dataclass(frozen=True)
class LLMTextDelta:
    """模型产生的一段可直接展示给用户的文本。"""

    text: str


@dataclass(frozen=True)
class LLMCompleted:
    """模型流的唯一完成事件，包含拼装后的完整响应。"""

    response: LLMResponse


LLMStreamEvent = LLMTextDelta | LLMCompleted


class BaseLLM(ABC):
    @abstractmethod
    async def stream(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict],
        *,
        deadline: Deadline,
    ) -> AsyncIterator[LLMStreamEvent]:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError


class ModelError(ExecutionError):
    """模型供应商无关的失败。"""

    error_code = "model_unavailable"
    retryable = False
    breaker_failure = False


class ModelTimeoutError(ModelError):
    error_code = "model_timeout"
    retryable = True
    breaker_failure = True


class ModelRateLimitError(ModelError):
    error_code = "model_rate_limited"
    retryable = True
    breaker_failure = True


class ModelUnavailableError(ModelError):
    error_code = "model_unavailable"
    retryable = True
    breaker_failure = True


class ModelRequestError(ModelError):
    error_code = "model_request_error"


class ModelResponseError(ModelError):
    error_code = "model_response_error"


class ModelCircuitOpenError(ModelError):
    error_code = "model_circuit_open"


class DeepSeekLLM(BaseLLM):
    """使用 AsyncOpenAI 的 DeepSeek 客户端，带有界重试和熔断。"""

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        attempt_timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if attempt_timeout <= 0:
            raise ValueError("attempt_timeout 必须大于 0")
        self._model = model
        self.attempt_timeout = attempt_timeout
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_timeout,
        )
        self._client = (
            client
            if client is not None
            else AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=attempt_timeout,
                # 重试只由本适配器管理，避免和 SDK 次数叠加。
                max_retries=0,
            )
        )

    async def stream(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict],
        *,
        deadline: Deadline,
    ) -> AsyncIterator[LLMStreamEvent]:
        started_at = time.monotonic()
        try:
            permit = self.circuit_breaker.acquire()
        except CircuitOpenError as exc:
            raise ModelCircuitOpenError(str(exc)) from exc

        permit_finalized = False
        emitted_text = False
        try:
            for attempt in range(1, self.retry_policy.max_attempts + 1):
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_call_parts: dict[int, dict[str, str]] = {}
                prompt_tokens = 0
                completion_tokens = 0
                finish_reason = "unknown"

                try:
                    async for chunk in self._request_stream_once(
                        messages,
                        tools,
                        deadline=deadline,
                    ):
                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            prompt_tokens = (
                                getattr(usage, "prompt_tokens", None) or 0
                            )
                            completion_tokens = (
                                getattr(usage, "completion_tokens", None) or 0
                            )

                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = getattr(choice, "delta", None)
                        if delta is None:
                            continue

                        text = getattr(delta, "content", None) or ""
                        if text:
                            content_parts.append(text)
                            emitted_text = True
                            yield LLMTextDelta(text=text)

                        reasoning = (
                            getattr(delta, "reasoning_content", None) or ""
                        )
                        if reasoning:
                            reasoning_parts.append(reasoning)

                        self._accumulate_tool_calls(
                            tool_call_parts,
                            getattr(delta, "tool_calls", None) or [],
                        )

                        if getattr(choice, "finish_reason", None):
                            finish_reason = choice.finish_reason

                    parsed = LLMResponse(
                        content="".join(content_parts),
                        reasoning_content="".join(reasoning_parts),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        finish_reason=finish_reason,
                        tool_calls=self._build_tool_calls(tool_call_parts),
                    )
                except RunDeadlineExceeded:
                    raise
                except Exception as exc:
                    error = self._map_error(exc)
                    if (
                        emitted_text
                        or not error.retryable
                        or attempt >= self.retry_policy.max_attempts
                    ):
                        if error.breaker_failure:
                            self.circuit_breaker.record_failure(permit)
                        else:
                            self.circuit_breaker.release_cancelled(permit)
                        permit_finalized = True
                        raise error from exc

                    delay = self.retry_policy.delay(
                        attempt,
                        retry_after=self._retry_after(exc),
                    )
                    await deadline.sleep(delay)
                    continue

                self.circuit_breaker.record_success(permit)
                permit_finalized = True
                yield LLMCompleted(
                    response=replace(
                        parsed,
                        attempts=attempt,
                        duration_ms=round(
                            (time.monotonic() - started_at) * 1000,
                            3,
                        ),
                    )
                )
                return
        finally:
            # 消费方取消或提前关闭异步生成器时，释放尚未结算的熔断许可。
            if not permit_finalized:
                self.circuit_breaker.release_cancelled(permit)

        raise AssertionError("模型重试循环未返回结果")

    async def aclose(self) -> None:
        """关闭应用生命周期内复用的异步 HTTP 客户端。"""
        await self._client.close()

    async def _request_stream_once(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict],
        *,
        deadline: Deadline,
    ) -> AsyncIterator[Any]:
        remaining = deadline.require_remaining()
        timeout = min(self.attempt_timeout, remaining)
        try:
            # SDK timeout 约束连接和相邻数据块读取；外层 deadline 约束整轮流。
            async with asyncio.timeout(remaining):
                response_stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=list(messages),
                    tools=list(tools) or None,
                    stream=True,
                    stream_options={"include_usage": True},
                    timeout=timeout,
                )
                async with response_stream:
                    async for chunk in response_stream:
                        yield chunk
        except TimeoutError as exc:
            if deadline.remaining() <= 0:
                raise RunDeadlineExceeded() from exc
            raise ModelTimeoutError("模型单次请求超时") from exc

    @staticmethod
    def _map_error(error: Exception) -> ModelError:
        if isinstance(error, ModelError):
            return error
        if isinstance(error, APITimeoutError):
            return ModelTimeoutError(str(error))
        if isinstance(error, (APIConnectionError, InternalServerError)):
            return ModelUnavailableError(str(error))
        if isinstance(error, RateLimitError):
            return ModelRateLimitError(str(error))
        if isinstance(error, APIStatusError):
            if error.status_code == 429:
                return ModelRateLimitError(str(error))
            if error.status_code >= 500:
                return ModelUnavailableError(str(error))
            return ModelRequestError(str(error))
        if isinstance(error, (json.JSONDecodeError, KeyError, IndexError)):
            return ModelResponseError(f"模型响应格式无效: {error}")
        return ModelRequestError(str(error))

    @staticmethod
    def _retry_after(error: Exception) -> float | None:
        if not isinstance(error, APIStatusError):
            return None
        value = error.response.headers.get("retry-after")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    @staticmethod
    def _accumulate_tool_calls(
        parts: dict[int, dict[str, str]],
        deltas: Sequence[Any],
    ) -> None:
        """按 index 拼装被模型流拆分的工具名称、ID 和 JSON 参数。"""
        for delta in deltas:
            index = int(delta.index)
            current = parts.setdefault(
                index,
                {"id": "", "name": "", "arguments": ""},
            )
            delta_id = getattr(delta, "id", None)
            if delta_id:
                current["id"] = DeepSeekLLM._merge_fragment(
                    current["id"],
                    delta_id,
                )

            function = getattr(delta, "function", None)
            if function is None:
                continue
            function_name = getattr(function, "name", None)
            if function_name:
                current["name"] = DeepSeekLLM._merge_fragment(
                    current["name"],
                    function_name,
                )
            if getattr(function, "arguments", None):
                current["arguments"] += function.arguments

    @staticmethod
    def _merge_fragment(current: str, incoming: str) -> str:
        """兼容字段只在首块出现、重复出现或按前缀分片的供应商。"""
        if not current:
            return incoming
        if incoming == current or current.endswith(incoming):
            return current
        if incoming.startswith(current):
            return incoming
        return current + incoming

    @staticmethod
    def _build_tool_calls(
        parts: dict[int, dict[str, str]],
    ) -> list[ToolRequest]:
        tool_calls: list[ToolRequest] = []
        for _, part in sorted(parts.items()):
            if not part["id"] or not part["name"]:
                raise ModelResponseError("模型工具调用缺少 id 或 name")
            try:
                arguments = json.loads(part["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                raise ModelResponseError(
                    f"模型工具参数不是有效 JSON: {exc}"
                ) from exc
            if not isinstance(arguments, dict):
                raise ModelResponseError("模型工具参数必须是 JSON object")
            tool_calls.append(
                ToolRequest(
                    tool_call_id=part["id"],
                    name=part["name"],
                    arguments=arguments,
                )
            )
        return tool_calls


__all__ = [
    "BaseLLM",
    "DeepSeekLLM",
    "LLMCompleted",
    "LLMResponse",
    "LLMStreamEvent",
    "LLMTextDelta",
    "ModelCircuitOpenError",
    "ModelError",
    "ModelRateLimitError",
    "ModelRequestError",
    "ModelResponseError",
    "ModelTimeoutError",
    "ModelUnavailableError",
]
