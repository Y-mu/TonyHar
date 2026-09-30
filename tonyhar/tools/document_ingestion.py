"""把已经解析好的正文写入知识库。"""

import asyncio
import hashlib
from typing import Any

from tonyhar.rag.text_ingestion import TextIngestionService
from tonyhar.tooling import BaseTool, ToolPolicy, tool


@tool(policy=ToolPolicy(timeout_seconds=60.0))
class DocumentIngestionTool(BaseTool):
    """接收正文字符串，完成切块并写入知识库。"""

    name = "document_ingestion"
    description = (
        "把已经读取或抓取的正文字符串切块并写入知识库。"
        "不要传入文件路径，text 必须是完整正文。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "需要写入知识库的完整正文字符串。",
            },
            "filename": {
                "type": "string",
                "description": "文档显示名称，例如 manual.md 或 article.md。",
            },
            "document_id": {
                "type": "string",
                "description": "可选的稳定文档标识；省略时由来源和正文生成。",
            },
            "source_type": {
                "type": "string",
                "description": "内容来源类型，例如 file、web 或 text。",
                "default": "text",
            },
            "source_uri": {
                "type": "string",
                "description": "可选的来源文件路径或网页 URL。",
            },
        },
        "required": ["text", "filename"],
        "additionalProperties": False,
    }

    def __init__(self, service: TextIngestionService):
        self.service = service

    async def execute(
        self,
        text: str,
        filename: str,
        document_id: str | None = None,
        source_type: str = "text",
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        resolved_document_id = document_id or self._build_document_id(
            text=text,
            filename=filename,
            source_type=source_type,
            source_uri=source_uri,
        )
        metadata: dict[str, Any] = {"source_type": source_type}
        if source_uri:
            metadata["source_uri"] = source_uri

        context = await asyncio.to_thread(
            self.service.ingest,
            text,
            document_id=resolved_document_id,
            filename=filename,
            metadata=metadata,
        )
        return {
            "success": True,
            "document_id": resolved_document_id,
            "filename": filename,
            "chunk_count": len(context.chunks),
        }

    @staticmethod
    def _build_document_id(
        *,
        text: str,
        filename: str,
        source_type: str,
        source_uri: str | None,
    ) -> str:
        identity = source_uri or f"{filename}\0{text}"
        payload = f"{source_type}\0{identity}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()
