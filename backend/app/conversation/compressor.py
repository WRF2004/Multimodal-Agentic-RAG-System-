"""
Context compressor utilities for managing token budgets.
"""

import tiktoken
import structlog

logger = structlog.get_logger()


class TokenCounter:
    """Counts tokens using tiktoken (cl100k_base for GPT-4 family)."""

    def __init__(self, model: str = "gpt-4o"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])


class ContextCompressor:
    """Compresses context to fit within token budget while preserving key information."""

    def __init__(self, max_context_tokens: int = 8000, model: str = "gpt-4o"):
        self.max_tokens = max_context_tokens
        self.counter = TokenCounter(model)

    def compress_documents(self, documents: list, max_tokens: int = None) -> list:
        """Truncate document contents to fit within token budget."""
        budget = max_tokens or self.max_tokens
        result = []
        used = 0
        for doc in documents:
            doc_tokens = self.counter.count(doc.content)
            if used + doc_tokens <= budget:
                result.append(doc)
                used += doc_tokens
            else:
                remaining = budget - used
                if remaining > 50:
                    doc.content = self.counter.truncate_to_tokens(doc.content, remaining)
                    result.append(doc)
                break
        return result