"""TonyHar 应用的组合根。

这里只负责创建基础设施并连接各模块，不处理 CLI、HTTP 或业务流程。
"""

import os
from pathlib import Path

from tonyhar.agent import Agent
from tonyhar.agent.dispatcher import RunDispatcher
from tonyhar.agent.llm import DeepSeekLLM
from tonyhar.agent.user_intent_recognizer import (
    get_shared_intent_recognizer,
)
from tonyhar.conversation import ChatService, InMemorySessionStore
from tonyhar.rag.chroma_store_imp import ChromaStoreImp
from tonyhar.rag.document_service import DocumentService
from tonyhar.rag.file_reader import FileTextReader
from tonyhar.rag.parser import TxtParser
from tonyhar.rag.retriever import Retriever
from tonyhar.rag.splitter import HybridSplitter, SplitterConfig
from tonyhar.rag.text_ingestion import TextIngestionService
from tonyhar.tooling import ToolManager
from tonyhar.tools.calculator import CalculatorTool
from tonyhar.tools.file_ingestion import FileIngestionService, FileIngestionTool
from tonyhar.tools.knowledge_list import KnowledgeListTool
from tonyhar.tools.knowledge_search import KnowledgeSearchTool
from tonyhar.tools.spider_url import GetUrlTool


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "你是一个会使用工具的助手。需要计算时调用 calculator。"
    "如果系统提示文件已入库，请向用户说明入库结果。"
    "回答知识库问题时，只能依据 knowledge_search 返回的片段，"
    "没有检索到相关内容时应明确说明。"
)


def build_agent() -> Agent:
    """创建无会话状态的 Agent 及其基础设施依赖。"""
    llm = DeepSeekLLM(api_key=os.environ.get("DEEPSEEK_API_KEY"))
    store = ChromaStoreImp(
        database_name=str(PROJECT_ROOT / "data" / "chroma"),
        collection_name="documents",
    )
    document_service = DocumentService(store)
    retriever = Retriever(store)
    splitter = HybridSplitter(
        config=SplitterConfig(chunk_size=512, chunk_overlap=64)
    )
    text_ingestion = TextIngestionService(
        splitter=splitter,
        document_service=document_service,
    )
    file_ingestion = FileIngestionService(
        reader=FileTextReader(TxtParser()),
        text_ingestion=text_ingestion,
    )

    tool_manager = ToolManager(
        providers=[
            CalculatorTool,
            lambda: FileIngestionTool(file_ingestion),
            lambda: KnowledgeListTool(document_service),
            lambda: KnowledgeSearchTool(retriever),
            GetUrlTool,
        ],
    )

    return Agent(
        llm=llm,
        tool_manager=tool_manager,
        run_dispatcher=RunDispatcher(get_shared_intent_recognizer()),
    )


def build_chat_service() -> ChatService:
    """创建应用级聊天服务及进程内会话存储。"""
    return ChatService(
        agent=build_agent(),
        sessions=InMemorySessionStore(),
        system_prompt=SYSTEM_PROMPT,
        max_turns=20,
        run_timeout_seconds=90.0,
    )
