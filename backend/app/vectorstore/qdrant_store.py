"""Qdrant vector store implementation."""

import uuid
import structlog
from typing import Optional

from app.core.interfaces import VectorStoreInterface, Document
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.VECTORSTORE, "qdrant")
class QdrantVectorStore(VectorStoreInterface):

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "documents",
        vector_size: int = 1536,
        **kwargs
    ):
        from qdrant_client import QdrantClient, models
        self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        self._models = models

        # Create collection if not exists
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        logger.info("qdrant_initialized", collection=collection_name, url=url)

    async def add_documents(self, documents: list[Document]) -> list[str]:
        from qdrant_client.models import PointStruct
        points = []
        ids = []
        for doc in documents:
            point_id = doc.id or str(uuid.uuid4())
            ids.append(point_id)
            payload = {**doc.metadata, "content": doc.content}
            points.append(PointStruct(
                id=point_id,
                vector=doc.embedding,
                payload=payload,
            ))
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )
        return ids

    async def search(
        self, query_embedding: list[float], top_k: int = 10, filters: Optional[dict] = None
    ) -> list[Document]:
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    self._models.FieldCondition(
                        key=key,
                        match=self._models.MatchValue(value=value)
                    )
                )
            query_filter = self._models.Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            Document(
                id=str(hit.id),
                content=hit.payload.get("content", ""),
                metadata={k: v for k, v in hit.payload.items() if k != "content"},
                score=hit.score,
            )
            for hit in results
        ]

    async def delete(self, document_ids: list[str]) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=self._models.PointIdsList(points=document_ids),
        )

    async def get_collection_stats(self) -> dict:
        info = self._client.get_collection(self._collection_name)
        return {
            "count": info.points_count,
            "name": self._collection_name,
            "status": info.status.value,
        }