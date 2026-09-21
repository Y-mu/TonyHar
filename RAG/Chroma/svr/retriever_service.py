from typing import Optional

from RAG.Chroma.model.RAGContext import RAGContext
from RAG.Chroma.model.SearchResult import SearchResult
from RAG.Chroma.store.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, context: RAGContext) -> None:
        self._vector_store = vector_store
        self._context = context

    def retrieve(self, 
                 query: str, 
                 top_k: Optional[int] = None,
                 score_threshold: Optional[float] = None,
                 filters: Optional[dict] = None) -> list[SearchResult]:
        config = self._context.retrieval
        results = self._vector_store.search(
            query=query,
            top_k=top_k if top_k is not None else config.top_k,
            filters=filters if filters is not None else config.filters,
        )

        threshold = score_threshold if score_threshold is not None else config.score_threshold
        filtered = (
            results
            if threshold is None
            else [result for result in results if result.score >= threshold]
        )
        self._context.recording_context.record("retrieval_query", query)
        self._context.recording_context.record("retrieval_results", filtered)
        return filtered
