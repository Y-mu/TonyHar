"""面向对象的 Chroma 基础设施适配层。"""
from tonyhar.model_config import (
    DEFAULT_EMBEDDING_MODEL_CONFIG,
    EmbeddingModelConfig,
    PROJECT_ROOT,
)

import os
import re
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import chromadb
import torch
from chromadb.utils import embedding_functions

from .vector_store_base import Chunk, DocumentSummary, VectorStore

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
        self._embedding_function: Any | None = None
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
    def embedding_function(self) -> Any:
        """首次检索或入库时才加载 embedding 模型。"""
        if self._embedding_function is None:
            self._embedding_function = self._create_embedding_function()
        return self._embedding_function

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

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """同时保留中文 unigram/bigram 与英文数字词，适合短查询。"""
        normalized = text.lower().strip()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
        chinese = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
        tokens.extend("".join(chinese[index:index + 2]) for index in range(len(chinese) - 1))
        return tokens

    def keyword_search(self, query: str, top_k: int) -> Sequence[Chunk]:
        """在 Chroma 已存正文上执行轻量 BM25 词法检索。"""
        if top_k <= 0:
            return []
        result = self.collection.get(include=["metadatas", "documents"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        rows: list[tuple[float, Chunk]] = []
        doc_tokens = [self._tokenize(document or "") for document in documents]
        avgdl = sum(len(tokens) for tokens in doc_tokens) / max(len(doc_tokens), 1)
        document_frequency: dict[str, int] = {}
        for tokens in doc_tokens:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total_docs = len(doc_tokens)
        for chunk_id, text, metadata, tokens in zip(ids, documents, metadatas, doc_tokens):
            metadata = metadata or {}
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            length = len(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                score += idf * (frequency * 2.0) / (
                    frequency + 1.5 * (0.25 + 0.75 * length / max(avgdl, 1.0))
                )
            if score <= 0:
                continue
            rows.append((score, Chunk(
                id=chunk_id,
                document_id=str(metadata.get("document_id", "")),
                filename=str(metadata.get("filename", "")),
                text=text or "",
                chunk_index=int(metadata.get("chunk_index", 0)),
                title=str(metadata.get("title", "")),
                page=metadata.get("page"),
                metadata=metadata,
            )))
        rows.sort(key=lambda row: (-row[0], row[1].id))
        return [chunk for _, chunk in rows[:top_k]]

    def list_documents(self) -> Sequence[DocumentSummary]:
        """无需加载 embedding 模型即可读取现有文档目录。"""
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return []

        result = collection.get(include=["metadatas"])
        documents: dict[str, DocumentSummary] = {}
        for metadata in result.get("metadatas") or []:
            metadata = metadata or {}
            document_id = str(metadata.get("document_id", ""))
            filename = str(metadata.get("filename", ""))
            key = document_id or filename
            if not key:
                continue
            documents.setdefault(
                key,
                DocumentSummary(
                    document_id=document_id,
                    filename=filename,
                ),
            )
        return sorted(documents.values(), key=lambda item: item.filename)
