from typing import Any, Dict

from chroma_config import (
    COLLECTION_NAME,
    DATABASE_NAME,
    create_client,
    create_embedding_function,
)


def search(query: str, n_results: int = 3) -> Dict[str, Any]:
    """Query an existing ChromaDB collection."""
    client = create_client(DATABASE_NAME)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=create_embedding_function(),
    )
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )
