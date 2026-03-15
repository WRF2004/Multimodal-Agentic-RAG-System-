"""BGE Reranker implementation."""

import asyncio
import structlog
from typing import Optional

from app.core.interfaces import RerankerInterface, Document
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.RERANKER, "bge")
class BGEReranker(RerankerInterface):

    def __init__(self, model_path: str = "BAAI/bge-reranker-v2-m3", **kwargs):
        self._model = None
        self._model_path = model_path

    def _load_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(self._model_path, use_fp16=True)
                logger.info("bge_reranker_loaded", model=self._model_path)
            except ImportError:
                logger.error("FlagEmbedding not installed. pip install FlagEmbedding")
                raise

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 5
    ) -> list[Document]:
        if not documents:
            return []
        self._load_model()
        pairs = [[query, doc.content] for doc in documents]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, lambda: self._model.compute_score(pairs, normalize=True)
        )
        if isinstance(scores, float):
            scores = [scores]

        for doc, score in zip(documents, scores):
            doc.score = float(score)

        ranked = sorted(documents, key=lambda d: d.score, reverse=True)
        return ranked[:top_k]


@register_component(ComponentRegistry.RERANKER, "cohere")
class CohereReranker(RerankerInterface):

    def __init__(
        self,
        api_key: str = "",
        model: str = "rerank-multilingual-v3.0",
        **kwargs
    ):
        self.api_key = api_key
        self.model = model

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 5
    ) -> list[Document]:
        if not documents:
            return []
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.cohere.com/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [doc.content for doc in documents],
                    "top_n": top_k,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", []):
            idx = item["index"]
            doc = documents[idx]
            doc.score = item["relevance_score"]
            results.append(doc)
        return sorted(results, key=lambda d: d.score, reverse=True)


@register_component(ComponentRegistry.RERANKER, "none")
class NoReranker(RerankerInterface):
    """Pass-through reranker that does nothing."""

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 5
    ) -> list[Document]:
        return documents[:top_k]