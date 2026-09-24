"""工具接口定义。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """执行工具并返回可序列化结果。"""
        raise NotImplementedError

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
