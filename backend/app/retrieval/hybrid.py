"""
Hybrid retriever combining dense and sparse retrieval with score fusion.
Supports Reciprocal Rank Fusion (RRF) and weighted linear combination.
"""

import structlog
from typing import Optional
from collections import defaultdict

from app.core.interfaces import RetrieverInterface, RetrievalResult, Document
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.RETRIEVER, "hybrid")
class HybridRetriever(RetrieverInterface):

    def __init__(
        self,
        dense_retriever: RetrieverInterface,
        sparse_retriever: RetrieverInterface,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        fusion_method: str = "rrf",  # "rrf" | "weighted"
        rrf_k: int = 60,
        **kwargs
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    async def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[dict] = None
    ) -> RetrievalResult:
        import asyncio
        dense_result, sparse_result = await asyncio.gather(
            self.dense.retrieve(query, top_k=top_k * 2, filters=filters),
            self.sparse.retrieve(query, top_k=top_k * 2, filters=filters),
        )

        if self.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(dense_result, sparse_result, top_k)
        else:
            fused = self._weighted_fusion(dense_result, sparse_result, top_k)

        return RetrievalResult(
            documents=fused,
            query=query,
            strategy_used="hybrid",
            scores=[d.score for d in fused],
            metadata={
                "dense_count": len(dense_result.documents),
                "sparse_count": len(sparse_result.documents),
                "fusion_method": self.fusion_method,
            }
        )

    def _reciprocal_rank_fusion(
        self,
        dense_result: RetrievalResult,
        sparse_result: RetrievalResult,
        top_k: int
    ) -> list[Document]:
        """RRF: score = sum(1 / (k + rank_i)) for each result list."""
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense_result.documents):
            key = doc.id or doc.content[:100]
            scores[key] += self.dense_weight / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(sparse_result.documents):
            key = doc.id or doc.content[:100]
            scores[key] += self.sparse_weight / (self.rrf_k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            doc.score = scores[key]
            results.append(doc)
        return results

    def _weighted_fusion(
        self,
        dense_result: RetrievalResult,
        sparse_result: RetrievalResult,
        top_k: int
    ) -> list[Document]:
        """Normalize scores to [0,1] and combine with weights."""
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        # Normalize dense scores
        dense_docs = dense_result.documents
        if dense_docs:
            max_s = max(d.score for d in dense_docs) or 1.0
            min_s = min(d.score for d in dense_docs)
            range_s = max_s - min_s or 1.0
            for doc in dense_docs:
                key = doc.id or doc.content[:100]
                norm = (doc.score - min_s) / range_s
                scores[key] += self.dense_weight * norm
                doc_map[key] = doc

        # Normalize sparse scores
        sparse_docs = sparse_result.documents
        if sparse_docs:
            max_s = max(d.score for d in sparse_docs) or 1.0
            min_s = min(d.score for d in sparse_docs)
            range_s = max_s - min_s or 1.0
            for doc in sparse_docs:
                key = doc.id or doc.content[:100]
                norm = (doc.score - min_s) / range_s
                scores[key] += self.sparse_weight * norm
                if key not in doc_map:
                    doc_map[key] = doc

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            doc.score = scores[key]
            results.append(doc)
        return results