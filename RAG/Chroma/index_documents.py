from typing import Any, Dict, List, Optional

from chroma_config import (
    COLLECTION_NAME,
    DATABASE_NAME,
    create_client,
    create_embedding_function,
)
from chunking import chunk_text


def index_documents(
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    chunk_size: int = 256,
    chunk_overlap: int = 30,
) -> int:
    """Chunk, embed, and store documents in ChromaDB."""
    if metadatas is not None and len(metadatas) != len(documents):
        raise ValueError("metadatas must have the same length as documents")

    chunks = []
    # 每个分块的唯一 ID：用于 ChromaDB 定位数据，upsert 重复运行时会更新同 ID 的记录。
    chunk_ids = []
    # 每个分块的附加信息：记录来源文档和分块位置，便于查询时过滤和溯源。
    chunk_metadatas = []

    for document_index, document in enumerate(documents):
        #固定token切割
        document_chunks = chunk_text(document, chunk_size, chunk_overlap)
        #索引基础原数据
        base_metadata = metadatas[document_index] if metadatas else {}

        for chunk_index, chunk in enumerate(document_chunks):
            chunks.append(chunk)
            chunk_ids.append(f"doc_{document_index}_chunk_{chunk_index}")
            chunk_metadatas.append(
                {
                    **base_metadata,
                    "document_index": document_index,
                    "chunk_index": chunk_index,
                }
            )

    if not chunks:
        return 0

    client = create_client(DATABASE_NAME)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=create_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        metadatas=chunk_metadatas,
    )
    return len(chunks)
