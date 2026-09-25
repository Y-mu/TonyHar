"""Web API 的请求和响应模型。"""

from typing import Annotated, Literal

from fastapi import Path
from pydantic import BaseModel, Field, field_validator


SessionId = Annotated[
    str,
    Path(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="会话 ID，只允许字母、数字、下划线和连字符",
    ),
]


class StreamMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message 不能为空")
        return value


class CreateSessionResponse(BaseModel):
    session_id: str


class ChatMessageResponse(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[ChatMessageResponse]
