"""会话领域模型。"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    """一个客户端会话的消息历史。"""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_turns: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if self.max_turns < 1:
            raise ValueError("max_turns 必须大于 0")
        self.session_id = self.session_id.strip()
        self.messages = deepcopy(self.messages)
        self._trim()

    @classmethod
    def create(
        cls,
        session_id: str,
        *,
        system_prompt: str = "",
        max_turns: int = 20,
    ) -> "Session":
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt.strip(),
            })
        return cls(
            session_id=session_id,
            messages=messages,
            max_turns=max_turns,
        )

    def snapshot(self) -> list[dict[str, Any]]:
        """返回可交给一次 Agent 运行独立修改的消息副本。"""
        return deepcopy(self.messages)

    def replace_messages(self, messages: list[dict[str, Any]]) -> None:
        self.messages = deepcopy(messages)
        self._trim()

    def clone(self) -> "Session":
        return Session(
            session_id=self.session_id,
            messages=self.messages,
            max_turns=self.max_turns,
        )

    def _trim(self) -> None:
        """按完整用户轮次裁剪，避免拆开 tool_call 与 tool result。"""
        system_messages = [
            message
            for message in self.messages
            if message.get("role") == "system"
        ]
        conversation_messages = [
            message
            for message in self.messages
            if message.get("role") != "system"
        ]

        turns: list[list[dict[str, Any]]] = []
        current_turn: list[dict[str, Any]] = []
        leading_messages: list[dict[str, Any]] = []
        for message in conversation_messages:
            if message.get("role") == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [message]
            elif current_turn:
                current_turn.append(message)
            else:
                leading_messages.append(message)
        if current_turn:
            turns.append(current_turn)

        if turns:
            kept_messages = [
                message
                for turn in turns[-self.max_turns:]
                for message in turn
            ]
        else:
            kept_messages = leading_messages

        self.messages = system_messages + kept_messages
