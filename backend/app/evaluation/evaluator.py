"""
RAG evaluation orchestrator.
Loads datasets, runs retrieval, computes metrics, and returns results.
"""

import json
import os
import structlog
from typing import Optional
from pathlib import Path

from app.core.interfaces import (
    RetrieverInterface, RerankerInterface, EvaluationMetricInterface
)
from app.core.registry import registry, ComponentRegistry

logger = structlog.get_logger()


class RAGEvaluator:
    """Orchestrates RAG evaluation over benchmark datasets."""

    def __init__(
        self,
        retriever: RetrieverInterface,
        reranker: Optional[RerankerInterface] = None,
        metrics: list[str] = None,
        datasets_dir: str = "./data/eval_datasets",
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.datasets_dir = datasets_dir
        self.metric_names = metrics or ["recall", "mrr", "ndcg", "precision", "hit_rate"]

    def _load_metric(self, name: str) -> EvaluationMetricInterface:
        return registry.create(ComponentRegistry.METRIC, name)

    def list_datasets(self) -> list[dict]:
        """List available evaluation datasets."""
        datasets = []
        ds_dir = Path(self.datasets_dir)
        if not ds_dir.exists():
            ds_dir.mkdir(parents=True, exist_ok=True)
            # Create a sample dataset
            self._create_sample_dataset(ds_dir)

        for f in ds_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                datasets.append({
                    "name": f.stem,
                    "path": str(f),
                    "num_queries": len(data.get("queries", [])),
                    "description": data.get("description", ""),
                })
            except Exception:
                continue
        return datasets

    def _create_sample_dataset(self, ds_dir: Path):
        """Create a sample evaluation dataset for demo purposes."""
        sample = {
            "description": "示例评测数据集 - 用于功能验证",
            "queries": [
                {
                    "query": "什么是RAG系统？",
                    "relevant_docs": ["doc_rag_intro", "doc_rag_overview"],
                },
                {
                    "query": "向量数据库有哪些？",
                    "relevant_docs": ["doc_vectordb"],
                },
                {
                    "query": "BM25算法的原理是什么？",
                    "relevant_docs": ["doc_bm25", "doc_sparse_retrieval"],
                },
            ]
        }
        with open(ds_dir / "sample.json", "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    async def evaluate(
        self,
        dataset_name: str,
        top_k: int = 10,
        use_reranker: bool = True,
        rerank_top_k: int = 5,
    ) -> dict:
        """Run evaluation on a dataset and return metric results."""
        # Load dataset
        dataset_path = Path(self.datasets_dir) / f"{dataset_name}.json"
        if not dataset_path.exists():
            return {"error": f"Dataset '{dataset_name}' not found"}

        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        queries_data = dataset.get("queries", [])
        if not queries_data:
            return {"error": "Dataset has no queries"}

        all_retrieved = []
        all_relevant = []
        query_results = []

        for qd in queries_data:
            query = qd["query"]
            relevant = qd.get("relevant_docs", [])

            # Retrieve
            result = await self.retriever.retrieve(query, top_k=top_k)
            docs = result.documents

            # Optionally rerank
            if use_reranker and self.reranker and docs:
                docs = await self.reranker.rerank(query, docs, top_k=rerank_top_k)

            retrieved_ids = [doc.id for doc in docs]
            all_retrieved.append(retrieved_ids)
            all_relevant.append(relevant)

            query_results.append({
                "query": query,
                "retrieved": retrieved_ids[:5],
                "relevant": relevant,
                "num_retrieved": len(retrieved_ids),
            })

        # Compute metrics
        metrics_results = {}
        for metric_name in self.metric_names:
            try:
                metric = self._load_metric(metric_name)
                score = metric.compute(all_retrieved, all_relevant)
                metrics_results[metric_name] = round(score, 4)
            except Exception as e:
                logger.error("metric_compute_error", metric=metric_name, error=str(e))
                metrics_results[metric_name] = None

        return {
            "dataset": dataset_name,
            "num_queries": len(queries_data),
            "top_k": top_k,
            "use_reranker": use_reranker,
            "metrics": metrics_results,
            "query_details": query_results,
        }