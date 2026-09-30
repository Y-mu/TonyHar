from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence,Optional, Any



@dataclass
class Chunk:
    id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int = 0
    content_with_weight: str = ""
    title: str = ""
    page: Optional[int] = None
    position: Optional[Any] = None
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    filename: str


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

    @abstractmethod
    def keyword_search(self, query: str, top_k: int) -> Sequence[Chunk]:
        """按词法相关性搜索 Chunk。"""
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> Sequence[DocumentSummary]:
        """列出已经写入的文档，不返回正文。"""
        raise NotImplementedError
