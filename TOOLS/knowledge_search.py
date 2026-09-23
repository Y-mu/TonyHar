"""Search the local knowledge base and return evidence chunks."""

from typing import Any

from core.tool import BaseTool
from rag.chroma_store_imp import ChromaStoreImp
from rag.retriever import Retriever


class KnowledgeSearchTool(BaseTool):
    """Expose retrieval as an Agent tool while keeping storage details hidden."""

    name = "knowledge_search"
    description = (
        "从已入库的知识库中检索与问题最相关的正文片段；"
        "回答文档或知识库相关问题前应调用此工具"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用于检索知识库的完整问题或关键词",
            },
            "top_k": {
                "type": "integer",
                "description": "返回片段数，默认 5，范围 1 到 20",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retriever: Retriever | None = None):
        # Delay model/database initialization until the first actual search.
        self._retriever = retriever

    def _get_retriever(self) -> Retriever:
        if self._retriever is None:
            store = ChromaStoreImp(
                database_name="data/chroma",
                collection_name="documents",
            )
            self._retriever = Retriever(store)
        return self._retriever

    def run(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("检索问题不能为空")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k 必须是整数")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k 必须在 1 到 20 之间")

        chunks = self._get_retriever().retrieve(query.strip(), top_k)
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
