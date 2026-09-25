"""模型端口与 DeepSeek 异步适配器。"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
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


class BaseLLM(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict],
        *,
        deadline: Deadline,
    ) -> LLMResponse:
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

    async def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict],
        *,
        deadline: Deadline,
    ) -> LLMResponse:
        started_at = time.monotonic()
        try:
            permit = self.circuit_breaker.acquire()
        except CircuitOpenError as exc:
            raise ModelCircuitOpenError(str(exc)) from exc

        try:
            for attempt in range(1, self.retry_policy.max_attempts + 1):
                try:
                    response = await self._request_once(
                        messages,
                        tools,
                        deadline=deadline,
                    )
                    parsed = self._parse_response(response)
                except asyncio.CancelledError:
                    self.circuit_breaker.release_cancelled(permit)
                    raise
                except RunDeadlineExceeded:
                    self.circuit_breaker.release_cancelled(permit)
                    raise
                except Exception as exc:
                    error = self._map_error(exc)
                    if (
                        not error.retryable
                        or attempt >= self.retry_policy.max_attempts
                    ):
                        if error.breaker_failure:
                            self.circuit_breaker.record_failure(permit)
                        else:
                            self.circuit_breaker.release_cancelled(permit)
                        raise error from exc

                    delay = self.retry_policy.delay(
                        attempt,
                        retry_after=self._retry_after(exc),
                    )
                    await deadline.sleep(delay)
                    continue

                self.circuit_breaker.record_success(permit)
                return replace(
                    parsed,
                    attempts=attempt,
                    duration_ms=round(
                        (time.monotonic() - started_at) * 1000,
                        3,
                    ),
                )
        except asyncio.CancelledError:
            self.circuit_breaker.release_cancelled(permit)
            raise
        except RunDeadlineExceeded:
            self.circuit_breaker.release_cancelled(permit)
            raise

        raise AssertionError("模型重试循环未返回结果")

    async def aclose(self) -> None:
        """关闭应用生命周期内复用的异步 HTTP 客户端。"""
        await self._client.close()

    async def _request_once(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict],
        *,
        deadline: Deadline,
    ) -> Any:
        remaining = deadline.require_remaining()
        timeout = min(self.attempt_timeout, remaining)
        bounded_by_run = remaining <= self.attempt_timeout
        try:
            async with asyncio.timeout(timeout):
                return await self._client.chat.completions.create(
                    model=self._model,
                    messages=list(messages),
                    tools=list(tools) or None,
                    timeout=timeout,
                )
        except TimeoutError as exc:
            if bounded_by_run and deadline.remaining() <= 0:
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
    def _parse_response(response: Any) -> LLMResponse:
        message = response.choices[0].message
        tool_calls: list[ToolRequest] = []
        for call in message.tool_calls or []:
            arguments = json.loads(call.function.arguments or "{}")
            tool_calls.append(
                ToolRequest(
                    tool_call_id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        usage = response.usage
        return LLMResponse(
            content=message.content or "",
            reasoning_content=(
                getattr(message, "reasoning_content", None) or ""
            ),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=response.choices[0].finish_reason or "unknown",
            tool_calls=tool_calls,
        )


__all__ = [
    "BaseLLM",
    "DeepSeekLLM",
    "LLMResponse",
    "ModelCircuitOpenError",
    "ModelError",
    "ModelRateLimitError",
    "ModelRequestError",
    "ModelResponseError",
    "ModelTimeoutError",
    "ModelUnavailableError",
]
