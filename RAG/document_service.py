"""文档入库应用服务。"""

from collections.abc import Sequence

from .vector_store_base import Chunk, VectorStore


class DocumentService:
    """负责把已经解析、切分好的 Chunk 写入向量存储。"""

    def __init__(self, vector_store: VectorStore):
        # 依赖抽象接口，不依赖 ChromaStoreImp，方便替换数据库。
        self.vector_store = vector_store

    @staticmethod
    def prepare_chunk(chunk: Chunk) -> Chunk:
        """生成入库所需的业务 metadata。"""
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
            metadata.setdefault("position", str(chunk.position))

        return Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            content_with_weight=chunk.content_with_weight,
            title=chunk.title,
            page=chunk.page,
            position=chunk.position,
            embedding=chunk.embedding,
            metadata=metadata,
            raw=chunk.raw,
        )

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """新增或更新一批文档切片。"""
        prepared_chunks = [self.prepare_chunk(chunk) for chunk in chunks]
        return self.vector_store.upsert(prepared_chunks)

    def delete_document(self, document_id: str) -> int:
        """删除一个文档对应的全部切片。"""
        return self.vector_store.delete_document(document_id)
