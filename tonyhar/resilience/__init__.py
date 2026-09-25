"""模型和工具共享的执行预算、重试与熔断原语。"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass


class ExecutionError(RuntimeError):
    """可以稳定映射到 Agent 终态协议的运行错误。"""

    error_code = "execution_error"


class RunDeadlineExceeded(ExecutionError):
    error_code = "run_timeout"

    def __init__(self, message: str = "Agent 运行超过总时间预算") -> None:
        super().__init__(message)


class CircuitOpenError(ExecutionError):
    error_code = "dependency_circuit_open"


@dataclass(frozen=True)
class Deadline:
    """基于单调时钟的整轮运行截止时间。"""

    expires_at: float

    @classmethod
    def after(cls, timeout_seconds: float) -> "Deadline":
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        return cls(time.monotonic() + timeout_seconds)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def require_remaining(self) -> float:
        remaining = self.remaining()
        if remaining <= 0:
            raise RunDeadlineExceeded()
        return remaining

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            self.require_remaining()
            return
        remaining = self.require_remaining()
        if delay >= remaining:
            raise RunDeadlineExceeded()
        await asyncio.sleep(delay)


@dataclass(frozen=True)
class RetryPolicy:
    """有限次数、带 full jitter 的指数退避策略。"""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 5.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于 0")
        if self.base_delay < 0:
            raise ValueError("base_delay 不能小于 0")
        if self.max_delay < 0:
            raise ValueError("max_delay 不能小于 0")

    def delay(self, failed_attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return max(0.0, retry_after)
        ceiling = min(
            self.max_delay,
            self.base_delay * (2 ** max(0, failed_attempt - 1)),
        )
        return random.uniform(0.0, ceiling) if self.jitter else ceiling


@dataclass(frozen=True)
class CircuitPermit:
    generation: int
    half_open_probe: bool = False


class CircuitBreaker:
    """支持并发调用和单 HALF_OPEN 探针的进程内熔断器。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold 必须大于 0")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout 必须大于 0")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_probe = False
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def acquire(self) -> CircuitPermit:
        with self._lock:
            state = self._state_locked()
            if state == "open":
                raise CircuitOpenError("依赖熔断器已打开，请稍后重试")
            if state == "half_open":
                if self._half_open_probe:
                    raise CircuitOpenError("依赖熔断器正在探测恢复")
                self._half_open_probe = True
                return CircuitPermit(self._generation, half_open_probe=True)
            return CircuitPermit(self._generation)

    def record_success(self, permit: CircuitPermit) -> None:
        with self._lock:
            if permit.generation != self._generation:
                return
            if self._opened_at is not None and not permit.half_open_probe:
                return
            self._failures = 0
            self._opened_at = None
            self._half_open_probe = False
            if permit.half_open_probe:
                self._generation += 1

    def record_failure(self, permit: CircuitPermit) -> None:
        with self._lock:
            if permit.generation != self._generation:
                return
            self._half_open_probe = False
            if permit.half_open_probe:
                self._open_new_generation_locked()
                return
            if self._opened_at is not None:
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open_new_generation_locked()

    def release_cancelled(self, permit: CircuitPermit) -> None:
        """取消不计失败，但必须释放 HALF_OPEN 探针名额。"""
        with self._lock:
            if (
                permit.generation == self._generation
                and permit.half_open_probe
            ):
                self._half_open_probe = False

    def _open_new_generation_locked(self) -> None:
        self._opened_at = time.monotonic()
        self._failures = self.failure_threshold
        self._half_open_probe = False
        self._generation += 1

    def _state_locked(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.recovery_timeout:
            return "half_open"
        return "open"


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitPermit",
    "Deadline",
    "ExecutionError",
    "RetryPolicy",
    "RunDeadlineExceeded",
]
