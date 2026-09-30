"""混合检索应用服务。"""

from collections.abc import Sequence

from .vector_store_base import Chunk, VectorStore


class Retriever:
    """融合 dense 向量召回与 sparse 词法召回。"""

    def __init__(self, vector_store: VectorStore):
        # 检索器只依赖 VectorStore 抽象，不感知底层数据库类型。
        self.vector_store = vector_store

    def retrieve_dense(self, query: str, top_k: int) -> Sequence[Chunk]:
        """返回 dense 向量召回结果。"""
        return self.vector_store.search(query, top_k)

    def retrieve_sparse(self, query: str, top_k: int) -> Sequence[Chunk]:
        """返回 sparse 词法召回结果。"""
        return self.vector_store.keyword_search(query, top_k)

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        *,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
    ) -> Sequence[Chunk]:
        """用 Reciprocal Rank Fusion 融合 dense 与 sparse 结果。

        两路检索使用较大的候选集，再按排名融合，避免不同后端分数
        尺度不一致导致某一路完全压制另一路。
        """
        if top_k < 1:
            return []
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("检索权重不能为负数")
        if dense_weight == 0 and sparse_weight == 0:
            raise ValueError("至少需要启用一种检索方式")

        candidate_k = max(top_k * 4, 20)
        dense = self.retrieve_dense(query, candidate_k) if dense_weight else []
        sparse = self.retrieve_sparse(query, candidate_k) if sparse_weight else []

        # RRF 对每路排名进行归一化，并以 chunk id 去重。
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        rrf_k = 60
        for weight, results in ((dense_weight, dense), (sparse_weight, sparse)):
            for rank, chunk in enumerate(results, start=1):
                scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (rrf_k + rank)
                chunks.setdefault(chunk.id, chunk)

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )
        return [chunks[chunk_id] for chunk_id in ranked_ids[:top_k]]
