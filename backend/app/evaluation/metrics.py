"""
Evaluation metrics for retrieval quality assessment.
All metrics implement EvaluationMetricInterface for extensibility.
"""

import numpy as np
from app.core.interfaces import EvaluationMetricInterface
from app.core.registry import register_component, ComponentRegistry


@register_component(ComponentRegistry.METRIC, "recall")
class RecallMetric(EvaluationMetricInterface):

    @property
    def name(self) -> str:
        return "recall"

    def compute(
        self,
        retrieved: list[list[str]],
        relevant: list[list[str]],
        **kwargs
    ) -> float:
        """
        Recall@K: fraction of relevant docs that were retrieved.
        retrieved: list of queries, each a list of retrieved doc IDs
        relevant: list of queries, each a list of relevant doc IDs
        """
        if not retrieved or not relevant:
            return 0.0
        recalls = []
        for ret, rel in zip(retrieved, relevant):
            if not rel:
                continue
            ret_set = set(ret)
            rel_set = set(rel)
            recalls.append(len(ret_set & rel_set) / len(rel_set))
        return float(np.mean(recalls)) if recalls else 0.0


@register_component(ComponentRegistry.METRIC, "precision")
class PrecisionMetric(EvaluationMetricInterface):

    @property
    def name(self) -> str:
        return "precision"

    def compute(
        self,
        retrieved: list[list[str]],
        relevant: list[list[str]],
        **kwargs
    ) -> float:
        if not retrieved or not relevant:
            return 0.0
        precisions = []
        for ret, rel in zip(retrieved, relevant):
            if not ret:
                precisions.append(0.0)
                continue
            ret_set = set(ret)
            rel_set = set(rel)
            precisions.append(len(ret_set & rel_set) / len(ret_set))
        return float(np.mean(precisions)) if precisions else 0.0


@register_component(ComponentRegistry.METRIC, "mrr")
class MRRMetric(EvaluationMetricInterface):
    """Mean Reciprocal Rank."""

    @property
    def name(self) -> str:
        return "mrr"

    def compute(
        self,
        retrieved: list[list[str]],
        relevant: list[list[str]],
        **kwargs
    ) -> float:
        if not retrieved or not relevant:
            return 0.0
        rrs = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            rr = 0.0
            for rank, doc_id in enumerate(ret, start=1):
                if doc_id in rel_set:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)
        return float(np.mean(rrs)) if rrs else 0.0


@register_component(ComponentRegistry.METRIC, "ndcg")
class NDCGMetric(EvaluationMetricInterface):
    """Normalized Discounted Cumulative Gain."""

    @property
    def name(self) -> str:
        return "ndcg"

    def compute(
        self,
        retrieved: list[list[str]],
        relevant: list[list[str]],
        **kwargs
    ) -> float:
        if not retrieved or not relevant:
            return 0.0
        ndcgs = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            dcg = 0.0
            for rank, doc_id in enumerate(ret, start=1):
                if doc_id in rel_set:
                    dcg += 1.0 / np.log2(rank + 1)
            # Ideal DCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), len(ret))))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(ndcgs)) if ndcgs else 0.0


@register_component(ComponentRegistry.METRIC, "hit_rate")
class HitRateMetric(EvaluationMetricInterface):
    """Hit Rate (at least one relevant doc in top-K)."""

    @property
    def name(self) -> str:
        return "hit_rate"

    def compute(
        self,
        retrieved: list[list[str]],
        relevant: list[list[str]],
        **kwargs
    ) -> float:
        if not retrieved or not relevant:
            return 0.0
        hits = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            hit = any(doc_id in rel_set for doc_id in ret)
            hits.append(1.0 if hit else 0.0)
        return float(np.mean(hits)) if hits else 0.0


@register_component(ComponentRegistry.METRIC, "map")
class MAPMetric(EvaluationMetricInterface):
    """Mean Average Precision."""

    @property
    def name(self) -> str:
        return "map"

    def compute(
        self,
        retrieved: list[list[str]],
        relevant: list[list[str]],
        **kwargs
    ) -> float:
        if not retrieved or not relevant:
            return 0.0
        aps = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            if not rel_set:
                continue
            hits = 0
            precision_sum = 0.0
            for rank, doc_id in enumerate(ret, start=1):
                if doc_id in rel_set:
                    hits += 1
                    precision_sum += hits / rank
            ap = precision_sum / len(rel_set)
            aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0