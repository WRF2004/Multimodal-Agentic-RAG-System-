"""OpenAI-compatible embedding provider."""

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.interfaces import EmbeddingInterface
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.EMBEDDING, "openai")
class OpenAIEmbedding(EmbeddingInterface):

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "text-embedding-3-small",
        _dimensions: int = 1536,
        batch_size: int = 64,
        **kwargs
    ):
        self.model = model
        self._dims = _dimensions
        self.batch_size = batch_size
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "no-key")
        logger.info("embedding_initialized", model=model)

    @property
    def dimensions(self) -> int:
        return self._dims

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # Clean empty strings
            batch = [t if t.strip() else "empty" for t in batch]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        result = await self.embed_texts([query])
        return result[0]


@register_component(ComponentRegistry.EMBEDDING, "sentence_transformer")
class SentenceTransformerEmbedding(EmbeddingInterface):

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        _dimensions: int = 384,
        **kwargs
    ):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)
        self._dims = _dimensions
        logger.info("st_embedding_initialized", model=model)

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, normalize_embeddings=True)
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        result = await self.embed_texts([query])
        return result[0]