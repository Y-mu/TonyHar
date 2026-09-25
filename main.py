"""聊天 Agent 入口。"""

import asyncio
import os

from tonyhar.agent import Agent
from tonyhar.agent.intent_planner import IntentPlanner
from tonyhar.agent.llm import DeepSeekLLM
from tonyhar.agent.user_intent_recognizer import (
    get_shared_intent_recognizer,
)
from tonyhar.conversation import ChatService, InMemorySessionStore
from tonyhar.tooling import ToolRegistry

from tonyhar.rag.chroma_store_imp import ChromaStoreImp
from tonyhar.rag.document_service import DocumentService
from tonyhar.rag.parser import TxtParser
from tonyhar.rag.retriever import Retriever
from tonyhar.rag.scheduler import DocumentScheduler
from tonyhar.rag.splitter import HybridSplitter, SplitterConfig
from tonyhar.tools.calculator import calculator
from tonyhar.tools.file_ingestion import create_file_ingestion_tool
from tonyhar.tools.knowledge_list import create_knowledge_list_tool
from tonyhar.tools.knowledge_search import create_knowledge_search_tool


SYSTEM_PROMPT = (
    "你是一个会使用工具的助手。需要计算时调用 calculator。"
    "如果系统提示文件已入库，请向用户说明入库结果。"
    "回答知识库问题时，只能依据 knowledge_search 返回的片段，"
    "没有检索到相关内容时应明确说明。"
)


def build_agent() -> Agent:
    llm = DeepSeekLLM(api_key=os.environ.get("DEEPSEEK_API_KEY"))
    store = ChromaStoreImp(
        database_name="data/chroma",
        collection_name="documents",
    )
    document_service = DocumentService(store)
    retriever = Retriever(store)
    scheduler = DocumentScheduler(
        parser=TxtParser(),
        splitter=HybridSplitter(
            config=SplitterConfig(chunk_size=512, chunk_overlap=64)
        ),
        document_service=document_service,
    )

    registry = ToolRegistry()
    registry.register_many(
        calculator,
        create_file_ingestion_tool(scheduler),
        create_knowledge_list_tool(document_service),
        create_knowledge_search_tool(retriever),
    )

    return Agent(
        llm=llm,
        tools=registry,
        intent_planner=IntentPlanner(get_shared_intent_recognizer()),
    )


def build_chat_service() -> ChatService:
    """装配无状态 Agent 和进程内会话存储。"""
    return ChatService(
        agent=build_agent(),
        sessions=InMemorySessionStore(),
        system_prompt=SYSTEM_PROMPT,
    )


async def main() -> None:
    chat_service = build_chat_service()
    session_id = "cli"
    await chat_service.create_session(session_id)

    try:
        while True:
            user_input = (await asyncio.to_thread(input, "你: ")).strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue
            result = await chat_service.invoke(session_id, user_input)
            if result.success:
                print("Agent:", result.answer)
            else:
                print("Agent error:", result.error_message or result.answer)
    finally:
        await chat_service.aclose()


if __name__ == "__main__":
    asyncio.run(main())
