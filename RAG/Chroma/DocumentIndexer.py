import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

try:
    from .chroma_config import get_collection
except ImportError:
    from chroma_config import get_collection
from RAG.Chroma.model.Chunk import Chunk

Metadata = Dict[str, Any]


@dataclass(frozen=True)
class IndexConfig:
    """索引服务配置；分块配置由 ChunkService 负责。"""

    batch_size: int = 256

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


class DocumentIndexer:
    """将 Chunk 流水线记录批量写入 Chroma。"""

    def __init__(self, config: Optional[IndexConfig] = None, collection_factory: Callable[[], Any] = get_collection) -> None:
        self.config = config or IndexConfig()
        self._collection_factory = collection_factory
        self._collection: Optional[Any] = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self._collection_factory()
        return self._collection

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """主索引入口：校验并批量 upsert Chunk。"""
        self._validate_chunks(chunks)
        total = 0
        for start in range(0, len(chunks), self.config.batch_size):
            batch = chunks[start : start + self.config.batch_size]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[self._metadata_for(chunk) for chunk in batch],
            )
            total += len(batch)
        return total

    def delete_document(self, document_id: str) -> None:
        """删除文档的全部 Chunk，重建索引前可调用。"""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        self.collection.delete(where={"document_id": document_id})

    @staticmethod
    def _metadata_for(chunk: Chunk) -> Metadata:
        metadata = dict(chunk.metadata)
        metadata.update({
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "chunk_index": chunk.chunk_index,
        })
        if chunk.title:
            metadata.setdefault("title", chunk.title)
        if chunk.page is not None:
            metadata.setdefault("page", chunk.page)
        if chunk.position is not None:
            metadata.setdefault("position", json.dumps(chunk.position, ensure_ascii=False))
        return metadata

    @staticmethod
    def _validate_chunks(chunks: Sequence[Chunk]) -> None:
        ids = []
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise TypeError("index_chunks expects Chunk objects")
            if not chunk.id.strip():
                raise ValueError("chunk.id must not be empty")
            if not chunk.text.strip():
                raise ValueError("chunk.text must not be empty: %s" % chunk.id)
            ids.append(chunk.id)
        if len(set(ids)) != len(ids):
            raise ValueError("chunk IDs must be unique")
