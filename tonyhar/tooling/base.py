"""工具接口和服务端执行策略。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


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


@dataclass(frozen=True)
class ToolDefinition:
    """工具对模型可见的静态定义。"""

    name: str
    description: str
    parameters: dict[str, Any]
    policy: ToolPolicy


class BaseTool(ABC):
    """所有业务工具必须实现的统一接口。"""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    policy: ClassVar[ToolPolicy] = ToolPolicy()
    __tool_definition__: ClassVar[ToolDefinition | None] = None

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """异步执行工具并返回可序列化结果。"""
        raise NotImplementedError

    def format_result(self, data: Any) -> str | None:
        """返回面向用户的展示文本；默认继续展示结构化 JSON。"""
        return None

    @property
    def definition(self) -> ToolDefinition:
        """返回工具的完整静态定义。"""
        declared = self.__dict__.get("__tool_definition__")
        if declared is None:
            declared = self.__class__.__dict__.get("__tool_definition__")
        if isinstance(declared, ToolDefinition):
            return declared
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            policy=self.policy,
        )

    def to_schema(self) -> dict:
        """转换为 OpenAI 兼容的 Function Calling schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.definition.name,
                "description": self.definition.description,
                "parameters": self.definition.parameters,
            },
        }
