"""Recursive character text splitter - splits on hierarchical separators."""

import uuid
from typing import Optional
from app.core.interfaces import ChunkerInterface, Document
from app.core.registry import register_component, ComponentRegistry


@register_component(ComponentRegistry.CHUNKER, "recursive")
class RecursiveChunker(ChunkerInterface):

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] = None,
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", "；", ";", " ", ""]

    async def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        if not text or not text.strip():
            return []
        raw_chunks = self._split_recursive(text, self.separators)
        documents = []
        for i, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():
                documents.append(Document(
                    id=str(uuid.uuid4()),
                    content=chunk_text.strip(),
                    metadata={
                        **(metadata or {}),
                        "chunk_index": i,
                        "chunk_strategy": "recursive",
                    }
                ))
        return documents

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._split_by_size(text)
        separator = separators[0]
        remaining_separators = separators[1:]
        if separator == "":
            return self._split_by_size(text)

        splits = text.split(separator) if separator else [text]
        good_splits = []
        current = ""
        for s in splits:
            candidate = current + separator + s if current else s
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good_splits.append(current)
                if len(s) > self.chunk_size:
                    good_splits.extend(self._split_recursive(s, remaining_separators))
                    current = ""
                else:
                    current = s
        if current:
            good_splits.append(current)
        return good_splits

    def _split_by_size(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks