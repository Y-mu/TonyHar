"""工具接口和服务端执行策略。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolPolicy:
    """由服务端注册、不会暴露给模型修改的工具执行策略。"""

    timeout_seconds: float = 10.0
    max_attempts: int = 1
    idempotent: bool = False
    parallel_safe: bool = False
    retry_base_delay: float = 0.25
    retry_max_delay: float = 2.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于 0")
        if self.max_attempts > 1 and not self.idempotent:
            raise ValueError("只有幂等工具可以配置重试")
        if self.retry_base_delay < 0 or self.retry_max_delay < 0:
            raise ValueError("工具重试延迟不能小于 0")


class RetryableToolError(RuntimeError):
    """工具可用此异常显式声明一次暂时性失败。"""


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict
    policy: ToolPolicy = ToolPolicy()

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """异步执行工具并返回可序列化结果。"""
        raise NotImplementedError

    def format_result(self, data: Any) -> str | None:
        """返回面向用户的展示文本；默认继续展示结构化 JSON。"""
        return None

    def to_schema(self) -> dict:
        """转换为 OpenAI 兼容的 Function Calling schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
