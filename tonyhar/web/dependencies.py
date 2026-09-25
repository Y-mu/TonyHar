"""FastAPI 依赖。"""

from fastapi import Request

from tonyhar.conversation import ChatService


def get_chat_service(request: Request) -> ChatService:
    """取得应用生命周期中创建的唯一 ChatService。"""
    return request.app.state.chat_service
