from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    name: str
    description: str
    input_schema: Dict[str, Any]

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError
