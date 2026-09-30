"""会话、消息和短期记忆领域模型。"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class TurnStatus(str, Enum):
    """一次用户轮次的持久化状态。"""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ConversationMessage:
    """持久化消息；运行时仍使用供应商兼容的字典消息。"""

    id: str
    turn_id: str
    role: str
    content: str
    status: TurnStatus = TurnStatus.COMPLETED
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.turn_id.strip():
            raise ValueError("消息 ID 和 turn_id 不能为空")
        if self.role not in {"user", "assistant", "tool"}:
            raise ValueError(f"不支持的消息角色: {self.role}")
        self.content = str(self.content)
        self.metadata = deepcopy(self.metadata)

    def to_llm(self) -> dict[str, Any]:
        """转换为模型运行时消息，不暴露持久化字段。"""
        return {
            "role": self.role,
            "content": self.content,
            **deepcopy(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "status": self.status.value,
            "metadata": deepcopy(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        return cls(
            id=str(data["id"]),
            turn_id=str(data["turn_id"]),
            role=str(data["role"]),
            content=str(data.get("content", "")),
            status=TurnStatus(str(data.get("status", TurnStatus.COMPLETED.value))),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(
                data.get("created_at")
                or datetime.now(timezone.utc).isoformat()
            ),
        )


@dataclass
class MemorySnapshot:
    """由旧轮次压缩得到的会话级工作记忆。"""

    summary: str = ""
    task_state: dict[str, Any] = field(default_factory=dict)
    covered_through_message_id: str | None = None
    version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "task_state": deepcopy(self.task_state),
            "covered_through_message_id": self.covered_through_message_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemorySnapshot":
        data = data or {}
        return cls(
            summary=str(data.get("summary", "")),
            task_state=dict(data.get("task_state") or {}),
            covered_through_message_id=data.get("covered_through_message_id"),
            version=int(data.get("version", 0)),
        )


@dataclass
class Session:
    """一个客户端会话的完整日志与短期记忆快照。"""

    session_id: str
    system_prompt: str = ""
    messages: list[ConversationMessage] = field(default_factory=list)
    memory: MemorySnapshot = field(default_factory=MemorySnapshot)
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if self.revision < 0:
            raise ValueError("revision 不能小于 0")
        self.session_id = self.session_id.strip()
        self.system_prompt = self.system_prompt.strip()
        self.messages = deepcopy(self.messages)
        self.memory = deepcopy(self.memory)

    @classmethod
    def create(cls, session_id: str, *, system_prompt: str = "") -> "Session":
        return cls(session_id=session_id, system_prompt=system_prompt)

    def append_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        status: TurnStatus,
        turn_id: str | None = None,
    ) -> None:
        """把一次运行新增的消息作为完整轮次写入日志。"""
        resolved_turn_id = turn_id or uuid4().hex
        for raw in messages:
            role = str(raw.get("role", ""))
            content = str(raw.get("content", ""))
            metadata = {
                key: deepcopy(value)
                for key, value in raw.items()
                if key not in {"role", "content"}
            }
            self.messages.append(ConversationMessage(
                id=uuid4().hex,
                turn_id=resolved_turn_id,
                role=role,
                content=content,
                status=status,
                metadata=metadata,
            ))

    def clone(self) -> "Session":
        return deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "messages": [message.to_dict() for message in self.messages],
            "memory": self.memory.to_dict(),
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=str(data["session_id"]),
            system_prompt=str(data.get("system_prompt", "")),
            messages=[
                ConversationMessage.from_dict(message)
                for message in data.get("messages", [])
            ],
            memory=MemorySnapshot.from_dict(data.get("memory")),
            revision=int(data.get("revision", 0)),
        )
