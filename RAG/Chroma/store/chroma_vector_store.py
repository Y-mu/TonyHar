import json
from typing import Any, Callable, Optional, Sequence

from RAG.Chroma.model.SearchResult import SearchResult

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
    def search(self,
               query: str,
               top_k: int,
               filters: Optional[dict] = None,
               ) -> list[SearchResult]:
        result = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filters,
            include=["metadatas", "documents", "distances"],
            
        )

        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]

        search_results = []

        for chunk_id, text, distance, metadata in zip(
            ids,
            documents,
            distances,
            metadatas,
        ):
            distance = float(distance)

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(
                        (metadata or {}).get("document_id", "")
                    ),
                    text=text or "",
                    distance=distance,
                    score=max(0.0, 1.0 - distance),
                    metadata=metadata or {},
                )
            )
        return search_results


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
