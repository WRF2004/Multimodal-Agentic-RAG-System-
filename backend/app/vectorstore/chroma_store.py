"""Chroma vector store implementation."""

import uuid
import structlog
from typing import Optional

from app.core.interfaces import VectorStoreInterface, Document
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.VECTORSTORE, "chroma")
class ChromaVectorStore(VectorStoreInterface):

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "documents",
        **kwargs
    ):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("chroma_initialized", collection=collection_name)

    async def add_documents(self, documents: list[Document]) -> list[str]:
        ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        self._collection.add(
            ids=ids,
            embeddings=[doc.embedding for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )
        return ids

    async def search(
        self, query_embedding: list[float], top_k: int = 10, filters: Optional[dict] = None
    ) -> list[Document]:
        where = filters if filters else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                documents.append(Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=1.0 - (results["distances"][0][i] if results["distances"] else 0),
                ))
        return documents

    async def delete(self, document_ids: list[str]) -> None:
        self._collection.delete(ids=document_ids)

    async def get_collection_stats(self) -> dict:
        return {"count": self._collection.count(), "name": self._collection.name}