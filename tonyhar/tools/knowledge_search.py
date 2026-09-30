"""知识库检索工具。"""

import asyncio

from typing import Any

from tonyhar.tooling import BaseTool, ToolPolicy, tool

from tonyhar.rag.retriever import Retriever


@tool(
    policy=ToolPolicy(
        timeout_seconds=20.0,
        max_attempts=2,
        idempotent=True,
        parallel_safe=True,
    ),
)
class KnowledgeSearchTool(BaseTool):
    """从知识库检索与问题最相关的正文片段。"""

    name = "knowledge_search"
    description = "从知识库检索与问题最相关的正文片段。回答文档问题前应使用此工具。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用于检索知识库的完整问题或关键词。",
            },
            "top_k": {
                "type": "integer",
                "description": "返回片段数，范围为 1 到 20。",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    async def execute(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("检索问题不能为空")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k 必须是整数")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k 必须在 1 到 20 之间")

        chunks = await asyncio.to_thread(
            self.retriever.retrieve_hybrid,
            query.strip(),
            top_k,
        )
        matches = [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "title": chunk.title,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        return {
            "query": query.strip(),
            "count": len(matches),
            "matches": matches,
        }
