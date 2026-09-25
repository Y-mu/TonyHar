"""TonyHar FastAPI 应用。"""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tonyhar.bootstrap import build_chat_service
from tonyhar.conversation import ChatService
from tonyhar.web.routes.chat import router as chat_router
from tonyhar.web.routes.health import router as health_router


ChatServiceFactory = Callable[[], ChatService]


def _cors_origins_from_environment() -> list[str]:
    configured = os.environ.get(
        "TONYHAR_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def create_app(
    service_factory: ChatServiceFactory = build_chat_service,
) -> FastAPI:
    """创建可运行、也可在测试中替换组合根的 ASGI 应用。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = service_factory()
        app.state.chat_service = service
        try:
            yield
        finally:
            await service.aclose()

    application = FastAPI(
        title="TonyHar API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_environment(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(chat_router, prefix="/api/v1")
    application.include_router(health_router)
    return application


app = create_app()
