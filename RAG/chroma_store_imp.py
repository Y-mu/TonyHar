"""面向对象的 Chroma 基础设施适配层。"""
from core.model_config import (
    DEFAULT_EMBEDDING_MODEL_CONFIG,
    EmbeddingModelConfig,
    PROJECT_ROOT,
)

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import chromadb
import torch
from chromadb.utils import embedding_functions

from .vector_store_base import Chunk, VectorStore

DEFAULT_DATABASE_NAME = "my_vector_db"
DEFAULT_COLLECTION_NAME = "my_documents"


class ChromaStoreImp(VectorStore):
    """封装 Chroma client、embedding function 和 collection。"""

    def __init__(self, database_name: str | Path = DEFAULT_DATABASE_NAME,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 model_config: EmbeddingModelConfig = DEFAULT_EMBEDDING_MODEL_CONFIG,
                 device: str | None = None):
        self.database_name = database_name
        self.collection_name = collection_name
        self.model_config = model_config
        self.model_name = model_config.model_name
        # 可通过 device 参数或 RAG_DEVICE 环境变量手动指定设备。
        self.device = (
            device
            or os.getenv(model_config.device_env_var)
            or self._get_device()
        )
        self.client = self._create_client()
        self.embedding_function = self._create_embedding_function()
        self._collection: Any | None = None

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def device_diagnostics() -> dict[str, object]:
        """返回设备诊断信息，便于解释为何回退 CPU。"""
        return {
            "torch_version": torch.__version__,
            "architecture": __import__("platform").machine(),
            "cuda_available": torch.cuda.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }

    def _create_client(self):
        path = Path(self.database_name)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(path))

    def _create_embedding_function(self):
        source = self.model_config.model_source
        print(f"嵌入模型: {self.model_name} (device={self.device})")
        if self.device == "cpu":
            diagnostics = self.device_diagnostics()
            print(f"设备诊断: {diagnostics}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=source,
            device=self.device,
            normalize_embeddings=True,
        )

    @property
    def collection(self) -> Any:
        """延迟创建并缓存 collection。"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """将已准备好的切片写入 Chroma，不参与业务 metadata 生成。"""
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[dict(chunk.metadata) for chunk in chunks],
        )
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """删除指定文档的全部切片。"""
        result = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )
        chunk_ids = result.get("ids", [])
        if not chunk_ids:
            return 0
        self.collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> Sequence[Chunk]:
        """按相似度查询，并转换为统一的 Chunk 对象。"""
        result = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filters,
            include=["metadatas", "documents"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        chunks = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            chunks.append(Chunk(
                id=chunk_id,
                document_id=str(metadata.get("document_id", "")),
                filename=str(metadata.get("filename", "")),
                text=text or "",
                chunk_index=int(metadata.get("chunk_index", 0)),
                title=str(metadata.get("title", "")),
                page=metadata.get("page"),
                metadata=metadata,
            ))
        return chunks
