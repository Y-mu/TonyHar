"""向量检索应用服务。"""

from collections.abc import Sequence

from .vector_store_base import Chunk, VectorStore


class Retriever:
    """负责查询向量存储并返回相关文档切片。"""

    def __init__(self, vector_store: VectorStore):
        # 检索器只依赖 VectorStore 抽象，不感知底层数据库类型。
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> Sequence[Chunk]:
        """返回与查询最相关的 top_k 个切片。"""
        return self.vector_store.search(query, top_k)
