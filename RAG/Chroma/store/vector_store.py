from abc import ABC, abstractmethod
from typing import Sequence

from ..model.Chunk import Chunk


class VectorStore(ABC):
    """向量存储统一接口。"""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """新增或更新 Chunk，返回成功写入数量。"""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """删除一个文档对应的所有 Chunk。"""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int) -> Sequence[Chunk]:
        """搜索与 query 相关的 top_k 个 Chunk。"""
        raise NotImplementedError
