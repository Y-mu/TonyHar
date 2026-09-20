from typing import Any, Dict

from RAG.Chroma.store.chroma_config import get_collection
# 查询已建立的向量索引


def search(query: str, n_results: int = 3) -> Dict[str, Any]:
    """Query an existing ChromaDB collection."""
    collection = get_collection()
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )
