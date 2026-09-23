"""将用户消息中引用的本地文本文件写入知识库。"""

import hashlib
from pathlib import Path
import re
from typing import Any

from core.parser import TxtParser
from core.tool import BaseTool
from rag.chroma_store_imp import ChromaStoreImp
from rag.document_service import DocumentService
from rag.pipeline_context import PipelineContext
from rag.scheduler import DocumentScheduler
from rag.splitter import SplitterConfig, TokenSplitter


class FileIngestionTool(BaseTool):
    """发现用户消息中的本地文本文件，并解析、切分和入库。"""

    name = "file_ingestion"
    description = "解析用户消息中引用的本地 .txt 或 .md 文件并写入知识库"
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "包含本地文件路径的完整用户消息",
            }
        },
        "required": ["message"],
        "additionalProperties": False,
    }
    SUPPORTED_SUFFIXES = frozenset({".txt", ".md"})
    MAX_FILES = 1_000

    def __init__(self, scheduler: DocumentScheduler | None = None):
        # 支持测试注入；生产环境仍然延迟加载 embedding 模型和 Chroma。
        self._scheduler = scheduler

    def _get_scheduler(self) -> DocumentScheduler:
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
    def _candidate_sources(message: str) -> list[Path]:
        """提取消息中的现有文件或目录路径。"""
        tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\S+)', message)
        sources: list[Path] = []
        for groups in tokens:
            token = next((item for item in groups if item), "").strip(
                "，。！？,;；"
            )
            if not token:
                continue
            path = Path(token).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                sources.append(path.resolve())

        # 中文指令经常不会在路径两侧加空格，例如“把orginalMd目录下的文件导入”。
        # 对工作目录直属文件和目录做名称匹配，补足普通空格分词无法覆盖的情况。
        try:
            workspace_entries = Path.cwd().iterdir()
        except OSError:
            workspace_entries = ()
        for entry in workspace_entries:
            if entry.name and entry.name in message:
                sources.append(entry.resolve())

        return list(dict.fromkeys(sources))

    @classmethod
    def _expand_files(cls, sources: list[Path]) -> list[Path]:
        """将文件和目录展开为支持入库的文件，目录采用递归扫描。"""
        files: list[Path] = []
        for source in sources:
            if source.is_file():
                files.append(source)
                continue
            if source.is_dir():
                files.extend(
                    path.resolve()
                    for path in sorted(source.rglob("*"))
                    if path.is_file()
                    and path.suffix.lower() in cls.SUPPORTED_SUFFIXES
                )
        return list(dict.fromkeys(files))

    def run(self, message: str) -> dict[str, Any]:
        """执行入库并返回适合工具调用消费的结构化结果。"""
        if not isinstance(message, str) or not message.strip():
            return {
                "success": False,
                "message": "入库消息不能为空",
                "files": [],
            }

        sources = self._candidate_sources(message)
        if not sources:
            return {
                "success": False,
                "message": "没有找到有效的本地文件或目录路径",
                "files": [],
            }

        paths = self._expand_files(sources)
        if not paths:
            return {
                "success": False,
                "message": "指定目录中没有可导入的 .txt 或 .md 文件",
                "files": [],
            }
        if len(paths) > self.MAX_FILES:
            return {
                "success": False,
                "message": (
                    f"待导入文件数 {len(paths)} 超过单次上限 "
                    f"{self.MAX_FILES}"
                ),
                "files": [],
            }

        files: list[dict[str, Any]] = []
        for path in paths:
            if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                files.append({
                    "filename": path.name,
                    "success": False,
                    "error": "暂不支持该文件类型",
                })
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
                files.append({
                    "filename": path.name,
                    "document_id": context.document_id,
                    "chunk_count": len(context.chunks),
                    "success": True,
                })
            except Exception as exc:
                files.append({
                    "filename": path.name,
                    "success": False,
                    "error": str(exc),
                })

        succeeded = sum(item["success"] for item in files)
        return {
            "success": succeeded == len(files),
            "message": f"成功入库 {succeeded}/{len(files)} 个文件",
            "files": files,
        }
