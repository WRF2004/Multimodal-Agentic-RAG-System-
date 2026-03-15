"""Fixed-size chunking with overlap."""

import uuid
from typing import Optional
from app.core.interfaces import ChunkerInterface, Document
from app.core.registry import register_component, ComponentRegistry


@register_component(ComponentRegistry.CHUNKER, "fixed")
class FixedChunker(ChunkerInterface):

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        if not text or not text.strip():
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Document(
                    id=str(uuid.uuid4()),
                    content=chunk_text,
                    metadata={
                        **(metadata or {}),
                        "chunk_index": len(chunks),
                        "chunk_strategy": "fixed",
                        "start_char": start,
                        "end_char": end,
                    }
                ))
            start += self.chunk_size - self.chunk_overlap
        return chunks