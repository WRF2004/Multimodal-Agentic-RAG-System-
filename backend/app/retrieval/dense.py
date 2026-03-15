"""Dense (vector) retriever."""

import structlog
from typing import Optional

from app.core.interfaces import (
    RetrieverInterface, RetrievalResult, EmbeddingInterface, VectorStoreInterface
)
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.RETRIEVER, "dense")
class DenseRetriever(RetrieverInterface):

    def __init__(
        self,
        embedding: EmbeddingInterface,
        vectorstore: VectorStoreInterface,
        **kwargs
    ):
        self.embedding = embedding
        self.vectorstore = vectorstore

    async def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[dict] = None
    ) -> RetrievalResult:
        query_embedding = await self.embedding.embed_query(query)
        documents = await self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )
        return RetrievalResult(
            documents=documents,
            query=query,
            strategy_used="dense",
            scores=[d.score for d in documents],
        )