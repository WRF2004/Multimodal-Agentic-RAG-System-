"""
Semantic chunking - splits based on embedding similarity between sentences.
Groups semantically similar consecutive sentences together.
"""

import uuid
import re
import numpy as np
import structlog
from typing import Optional

from app.core.interfaces import ChunkerInterface, Document, EmbeddingInterface
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.CHUNKER, "semantic")
class SemanticChunker(ChunkerInterface):

    def __init__(
        self,
        embedding: EmbeddingInterface = None,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.75,
        **kwargs
    ):
        self.embedding = embedding
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        pattern = r'(?<=[。！？.!?\n])\s*'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    async def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Document(
                id=str(uuid.uuid4()),
                content=text.strip(),
                metadata={**(metadata or {}), "chunk_index": 0, "chunk_strategy": "semantic"}
            )]

        # If no embedding model available, fall back to recursive
        if self.embedding is None:
            logger.warning("no_embedding_for_semantic_chunking, falling back to sentence grouping")
            return self._fallback_chunk(sentences, metadata)

        # Embed all sentences
        embeddings = await self.embedding.embed_texts(sentences)
        emb_array = np.array(embeddings)

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(emb_array) - 1):
            sim = np.dot(emb_array[i], emb_array[i + 1]) / (
                np.linalg.norm(emb_array[i]) * np.linalg.norm(emb_array[i + 1]) + 1e-8
            )
            similarities.append(sim)

        # Find split points where similarity drops below threshold
        chunks = []
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_len = current_len + len(next_sentence)

            if (sim < self.similarity_threshold and current_len >= self.min_chunk_size) or \
               next_len > self.max_chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [next_sentence]
                current_len = len(next_sentence)
            else:
                current_chunk.append(next_sentence)
                current_len = next_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        documents = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                documents.append(Document(
                    id=str(uuid.uuid4()),
                    content=chunk_text.strip(),
                    metadata={
                        **(metadata or {}),
                        "chunk_index": i,
                        "chunk_strategy": "semantic",
                    }
                ))
        return documents

    def _fallback_chunk(self, sentences: list[str], metadata: Optional[dict]) -> list[Document]:
        chunks = []
        current = []
        current_len = 0
        for s in sentences:
            if current_len + len(s) > self.max_chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(s)
            current_len += len(s)
        if current:
            chunks.append(" ".join(current))
        return [
            Document(
                id=str(uuid.uuid4()),
                content=c,
                metadata={**(metadata or {}), "chunk_index": i, "chunk_strategy": "semantic"}
            )
            for i, c in enumerate(chunks) if c.strip()
        ]