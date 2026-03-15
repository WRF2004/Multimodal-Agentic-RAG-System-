"""BM25 sparse retriever."""

import uuid
import math
import structlog
import numpy as np
from typing import Optional
from collections import Counter

from app.core.interfaces import RetrieverInterface, RetrievalResult, Document
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.RETRIEVER, "sparse")
class BM25Retriever(RetrieverInterface):
    """
    BM25 sparse retriever with in-memory index.
    Supports incremental document addition.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, **kwargs):
        self.k1 = k1
        self.b = b
        self._documents: list[Document] = []
        self._doc_freqs: Counter = Counter()
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._tokenized_docs: list[list[str]] = []
        self._n_docs: int = 0

    def add_documents(self, documents: list[Document]) -> None:
        for doc in documents:
            tokens = self._tokenize(doc.content)
            self._documents.append(doc)
            self._tokenized_docs.append(tokens)
            self._doc_lens.append(len(tokens))
            self._doc_freqs.update(set(tokens))
        self._n_docs = len(self._documents)
        self._avg_dl = sum(self._doc_lens) / self._n_docs if self._n_docs else 1.0

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc_tokens = self._tokenized_docs[doc_idx]
        doc_len = self._doc_lens[doc_idx]
        tf_map = Counter(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf_map:
                continue
            tf = tf_map[qt]
            df = self._doc_freqs.get(qt, 0)
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
            score += idf * numerator / denominator
        return score

    async def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[dict] = None
    ) -> RetrievalResult:
        if not self._documents:
            return RetrievalResult(documents=[], query=query, strategy_used="sparse")

        query_tokens = self._tokenize(query)
        scores = []
        for i in range(self._n_docs):
            if filters:
                match = all(
                    self._documents[i].metadata.get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    scores.append(-float("inf"))
                    continue
            scores.append(self._bm25_score(query_tokens, i))

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self._documents[idx]
            doc.score = float(scores[idx])
            results.append(doc)

        return RetrievalResult(
            documents=results,
            query=query,
            strategy_used="sparse",
            scores=[d.score for d in results],
        )