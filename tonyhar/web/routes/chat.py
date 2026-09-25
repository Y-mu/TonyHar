"""会话查询和 Agent 事件流接口。"""
import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette import EventSourceResponse

from tonyhar.conversation import ChatService
from tonyhar.web.dependencies import get_chat_service
from tonyhar.web.schemas import (
    ChatMessageResponse,
    CreateSessionResponse,
    SessionId,
    SessionResponse,
    StreamMessageRequest,
)
from tonyhar.web.sse import encode_agent_event, encode_stream_error

router = APIRouter(prefix="/sessions", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    service: ChatService = Depends(get_chat_service),
) -> CreateSessionResponse:
    session = await service.create_session()
    return CreateSessionResponse(session_id=session.session_id)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: SessionId,
    service: ChatService = Depends(get_chat_service),
) -> SessionResponse:
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在",
        )

    visible_messages = [
        ChatMessageResponse(
            role=message["role"],
            content=str(message.get("content", "")),
        )
        for message in session.messages
        if message.get("role") in {"user", "assistant"}
        and message.get("content")
    ]
    return SessionResponse(
        session_id=session.session_id,
        messages=visible_messages,
    )


@router.post("/{session_id}/messages/stream")
async def stream_message(
    session_id: SessionId,
    body: StreamMessageRequest,
    service: ChatService = Depends(get_chat_service),
) -> EventSourceResponse:
    async def generate():
        sequence = 0
        terminal_sent = False
        last_run_id = ""

        try:
            async for event in service.stream(session_id, body.message):
                sequence += 1
                last_run_id = event.run_id

                yield encode_agent_event(event, sequence)

                terminal_sent = event.type.value in {
                    "final_answer",
                    "run_failed",
                }

        except asyncio.CancelledError:
            # 客户端点击停止或关闭页面。
            # 必须继续向上传播，让 Agent 进入 CANCELLED，
            # ChatService 的 finally 才能保存状态并释放会话锁。
            raise

        except Exception:
            logger.exception("SSE stream failed")

            # 如果 Agent 已经发出终态，不再制造第二个终态事件。
            if not terminal_sent:
                sequence += 1
                yield encode_stream_error(
                    run_id=last_run_id or uuid4().hex,
                    session_id=session_id,
                    sequence=sequence,
                )

    return EventSourceResponse(
        generate(),
        ping=15,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
