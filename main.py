"""聊天 Agent 入口，并通过规则层自动处理消息中的文件。"""

import asyncio
import hashlib
import os
import re
from pathlib import Path

from core.agent import Agent
from core.llm import DeepSeekLLM
from core.parser import TxtParser
from core.tool import ToolRegistry
from rag.chroma_store_imp import ChromaStoreImp
from rag.document_service import DocumentService
from rag.pipeline_context import PipelineContext
from rag.scheduler import DocumentScheduler
from rag.splitter import SplitterConfig, TokenSplitter
from tools.calculator import CalculatorTool


class FileIngestionRule:
    """发现用户消息中的本地文本文件，并尝试解析、切分和入库。"""

    SUPPORTED_SUFFIXES = {".txt", ".md"}

    def __init__(self):
        self._scheduler: DocumentScheduler | None = None

    def _get_scheduler(self) -> DocumentScheduler:
        # 只有真正出现文件时才加载 embedding 模型和 Chroma。
        if self._scheduler is None:
            store = ChromaStoreImp(
                database_name="data/chroma",
                collection_name="documents",
            )
            self._scheduler = DocumentScheduler(
                parser=TxtParser(),
                splitter=TokenSplitter(
                    SplitterConfig(chunk_size=512, chunk_overlap=64)
                ),
                document_service=DocumentService(store),
            )
        return self._scheduler

    @staticmethod
    def _candidate_paths(message: str) -> list[Path]:
        """提取引号包裹或空格分隔的现有文件路径。"""
        tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\S+)', message)
        paths: list[Path] = []
        for groups in tokens:
            token = next((item for item in groups if item), "").strip("，。！？,;；")
            if not token:
                continue
            path = Path(token).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_file():
                paths.append(path.resolve())
        return list(dict.fromkeys(paths))

    def apply(self, message: str) -> list[str]:
        """执行文件规则并返回可提供给 Agent 的处理结果。"""
        results: list[str] = []
        for path in self._candidate_paths(message):
            if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                results.append(f"文件 {path.name} 暂不支持解析")
                continue
            try:
                context = PipelineContext(
                    document_id=hashlib.sha1(
                        str(path).encode("utf-8")
                    ).hexdigest(),
                    filename=path.name,
                    binary=path.read_bytes(),
                )
                self._get_scheduler().run(context)
                results.append(
                    f"文件 {path.name} 已解析并入库，共 {len(context.chunks)} 个切片"
                )
            except Exception as exc:
                results.append(f"文件 {path.name} 解析或入库失败：{exc}")
        return results


def build_agent() -> Agent:
    llm = DeepSeekLLM(api_key=os.environ.get("DEEPSEEK_API_KEY"))
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return Agent(
        llm=llm,
        tools=registry,
        system_prompt=(
            "你是一个会使用工具的助手。需要计算时调用 calculator。"
            "如果系统提示文件已入库，请向用户说明入库结果。"
        ),
    )

async def main() -> None:
    agent = build_agent()
    file_rule = FileIngestionRule()

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        rule_results = file_rule.apply(user_input)
        agent_input = user_input
        if rule_results:
            agent_input += "\n\n[规则层处理结果]\n" + "\n".join(rule_results)

        print("Agent:", await agent.run(agent_input))


if __name__ == "__main__":
    asyncio.run(main())
