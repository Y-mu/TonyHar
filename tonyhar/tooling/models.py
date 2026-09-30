"""与模型供应商无关的工具调用模型。"""

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class ToolRequest:
    """一次工具调用的标准输入。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("工具名称不能为空")
        if not isinstance(self.arguments, dict):
            raise TypeError("工具参数必须是 dict")
        if self.tool_call_id is not None and not isinstance(
            self.tool_call_id,
            str,
        ):
            raise TypeError("tool_call_id 必须是 str 或 None")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "arguments", dict(self.arguments))

    def to_openai(self) -> dict[str, Any]:
        """转换为 OpenAI 兼容的 assistant tool_call。"""
        if not self.tool_call_id:
            raise ValueError("写入消息历史前必须提供 tool_call_id")
        return {
            "id": self.tool_call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的标准输出。"""

    name: str
    success: bool
    tool_call_id: str | None = None
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    attempts: int = 1
    duration_ms: float = 0.0
    display_content: str | None = None

    @property
    def content(self) -> str:
        """序列化为可直接写入 LLM tool message 的 JSON。"""
        payload = self.data if self.success else {
            "success": False,
            "error": {
                "code": self.error_code or "tool_execution_error",
                "message": self.error_message or "工具执行失败",
            },
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def content_for_model(self, max_chars: int) -> str:
        """返回受限的模型上下文内容，完整 data 仍用于业务和展示。"""
        if max_chars < 256:
            raise ValueError("max_chars 不能小于 256")
        content = self.content
        if len(content) <= max_chars:
            return content
        low, high = 0, max_chars
        best = '{"truncated":true}'
        while low <= high:
            middle = (low + high) // 2
            candidate = json.dumps(
                {
                    "truncated": True,
                    "original_chars": len(content),
                    "preview": content[:middle],
                },
                ensure_ascii=False,
            )
            if len(candidate) <= max_chars:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "tool_call_id": self.tool_call_id,
            "data": self.data,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
        }
