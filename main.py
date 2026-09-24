"""聊天 Agent 入口。"""

import asyncio
import os

from core.agent import Agent
from core.intent_planner import IntentPlanner
from core.llm import DeepSeekLLM
from core.parser import TxtParser
from core.userIntentrecognizer import get_shared_intent_recognizer
from tooling import ToolRegistry

from rag.chroma_store_imp import ChromaStoreImp
from rag.document_service import DocumentService
from rag.retriever import Retriever
from rag.scheduler import DocumentScheduler
from rag.splitter import HybridSplitter, SplitterConfig
from tools.calculator import calculator
from tools.file_ingestion import create_file_ingestion_tool
from tools.knowledge_list import create_knowledge_list_tool
from tools.knowledge_search import create_knowledge_search_tool


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
        system_prompt=(
            "你是一个会使用工具的助手。需要计算时调用 calculator。"
            "如果系统提示文件已入库，请向用户说明入库结果。"
            "回答知识库问题时，只能依据 knowledge_search 返回的片段，"
            "没有检索到相关内容时应明确说明。"
        ),
    )

async def main() -> None:
    agent = build_agent()

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        print("Agent:", await agent.invoke(user_input))


if __name__ == "__main__":
    asyncio.run(main())
