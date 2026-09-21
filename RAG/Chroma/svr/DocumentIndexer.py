from dataclasses import dataclass
from typing import Optional, Sequence

from ..model.Chunk import Chunk
from ..store.vector_store import VectorStore


@dataclass(frozen=True)
class IndexConfig:
    """索引服务配置；分块配置由 ChunkService 负责。"""

    batch_size: int = 256

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


class DocumentIndexer:
    """将 Chunk 流水线记录批量写入 Chroma。"""

    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[IndexConfig] = None,
    ) -> None:
        self._vector_store = vector_store
        self.config = config or IndexConfig()


    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """主索引入口：校验并批量 upsert Chunk。"""
        self._validate_chunks(chunks)
        total = 0

        for start in range(0, len(chunks), self.config.batch_size):
            batch = chunks[start : start + self.config.batch_size]

            written_count = self._vector_store.upsert(batch)
            total += written_count

        return total

    def delete_document(self, document_id: str) -> int:
        """删除文档的全部 Chunk，重建索引前可调用。"""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        return self._vector_store.delete_document(document_id)

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
