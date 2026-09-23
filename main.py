"""聊天 Agent 入口。"""

import asyncio
import os

from core.agent import Agent
from core.llm import DeepSeekLLM
from core.userIntentrecognizer import get_shared_intent_recognizer
from core.tool import ToolRegistry
from tools.calculator import CalculatorTool
from tools.file_ingestion import FileIngestionTool
from tools.knowledge_list import KnowledgeListTool
from tools.knowledge_search import KnowledgeSearchTool


def build_agent() -> Agent:
    llm = DeepSeekLLM(api_key=os.environ.get("DEEPSEEK_API_KEY"))
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileIngestionTool())
    registry.register(KnowledgeListTool())
    registry.register(KnowledgeSearchTool())
    return Agent(
        llm=llm,
        tools=registry,
        intent_recognizer=get_shared_intent_recognizer(),
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
        print("Agent:", await agent.run(user_input))


if __name__ == "__main__":
    asyncio.run(main())
