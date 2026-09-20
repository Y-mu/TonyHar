import json
from typing import Any, Callable, Optional, Sequence

from ..model.Chunk import Chunk
from .chroma_config import get_collection
from .vector_store import VectorStore



class ChromaVectorStore(VectorStore):
    def __init__(self, collection_factory: Callable[[], Any] = get_collection) -> None:
        self._collection_factory = collection_factory
        self._collection: Optional[Any] = None
        
    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self._collection_factory()

        return self._collection

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                self._build_metadata(chunk)
                for chunk in chunks
            ],
        )

        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        result = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        chunk_ids = result.get("ids", [])

        if not chunk_ids:
            return 0

        self.collection.delete(ids=chunk_ids)
        return len(chunk_ids)
            
    @staticmethod
    def _build_metadata(chunk: Chunk) -> dict[str, Any]:
        metadata = dict(chunk.metadata)

        metadata.update(
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
            }
        )

        if chunk.title:
            metadata.setdefault("title", chunk.title)

        if chunk.page is not None:
            metadata.setdefault("page", chunk.page)

        if chunk.position is not None:
            metadata.setdefault(
                "position",
                json.dumps(
                    chunk.position,
                    ensure_ascii=False,
                ),
            )

        return metadata
