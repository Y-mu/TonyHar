"""TonyHar Web 应用入口。

本模块只负责 HTTP 应用的装配和生命周期管理，不承载会话或 Agent 业务逻辑：

- ``tonyhar.bootstrap`` 创建完整的 ``ChatService`` 依赖图；
- FastAPI lifespan 管理该服务的启动和关闭；
- routes 将 HTTP/SSE 请求适配到 ``ChatService``；
- CLI 和 Web 因此可以复用同一套业务服务与运行协议。

Uvicorn 默认通过模块级 ``app`` 启动应用；测试则调用 ``create_app`` 注入隔离的
``ChatService``，避免加载真实模型和向量数据库。
"""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tonyhar.bootstrap import build_chat_service
from tonyhar.conversation import ChatService
from tonyhar.web.routes.chat import router as chat_router
from tonyhar.web.routes.health import router as health_router


# 使用工厂而不是已创建的实例，确保 ChatService 在 ASGI lifespan 开始时创建，
# 并且测试可以注入轻量实现而不修改生产组合根。
ChatServiceFactory = Callable[[], ChatService]


def _cors_origins_from_environment() -> list[str]:
    """读取浏览器可访问 API 的来源白名单。

    本地开发需要同时支持 ``localhost`` 和 ``127.0.0.1``；生产环境应通过
    ``TONYHAR_CORS_ORIGINS`` 显式传入逗号分隔的前端来源，而不是使用通配符。
    """
    configured = os.environ.get(
        "TONYHAR_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def create_app(
    service_factory: ChatServiceFactory = build_chat_service,
) -> FastAPI:
    """创建可运行、也可在测试中替换依赖组合的 ASGI 应用。

    Args:
        service_factory: 创建应用级 ``ChatService`` 的工厂。生产环境使用
            ``build_chat_service``，测试可以注入不访问外部依赖的服务。

    Returns:
        已注册生命周期、中间件和路由的 FastAPI 应用。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 每个应用进程只创建一个 ChatService。它内部的 Agent、模型客户端、
        # SessionStore 和会话锁会被所有请求共享；不要在单个路由中重复装配。
        service = service_factory()

        # 路由通过 FastAPI dependency 从 app.state 取得服务，保持路由层只依赖
        # ChatService，而不直接接触 Agent、工具注册表或 SessionStore。
        app.state.chat_service = service
        try:
            yield
        finally:
            # 即使应用启动后异常退出，也必须关闭模型客户端的连接池等资源。
            await service.aclose()

    application = FastAPI(
        title="TonyHar API",
        version="1.0.0",
        lifespan=lifespan,
    )
    # 当前 Web API 只需要 GET 查询和 POST 命令/SSE。限制 method/header 范围
    # 可以减少不必要的跨域能力；项目没有基于 Cookie 的身份认证，因此关闭凭证。
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_environment(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    # 业务接口带稳定的版本前缀；健康检查不带版本，便于反向代理和部署平台探测。
    application.include_router(chat_router, prefix="/api/v1")
    application.include_router(health_router)
    return application


# Uvicorn 导入 ``tonyhar.web.app:app`` 时使用的生产应用实例。
app = create_app()
