from abc import ABC, abstractmethod
from typing import Any
import json

class BaseTool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema

    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...

    def to_schema(self) -> dict:
        """转成 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, args: dict) -> str:
        tool = self.get(name)
        if not tool:
            return f"Error: tool '{name}' not found"
        try:
            result = tool.run(**args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error: {e}"