"""
Evaluation API endpoints.
"""

import structlog
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.config import config_manager
from app.dependencies import factory

logger = structlog.get_logger()
router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])


class EvalRequest(BaseModel):
    session_id: str = "default"
    dataset_name: str
    top_k: int = 10
    use_reranker: bool = True
    rerank_top_k: int = 5


class EvalResponse(BaseModel):
    dataset: str
    num_queries: int
    metrics: dict
    query_details: list = []
    config_used: dict = {}


@router.get("/datasets")
async def list_datasets(session_id: str = "default"):
    """List available evaluation datasets."""
    cfg = config_manager.get_session_config(session_id)
    components = factory.build_all(cfg)
    evaluator = components["evaluator"]
    return evaluator.list_datasets()


@router.post("/run", response_model=EvalResponse)
async def run_evaluation(request: EvalRequest):
    """Run evaluation on a selected dataset."""
    cfg = config_manager.get_session_config(request.session_id)
    components = factory.build_all(cfg)
    evaluator = components["evaluator"]

    result = await evaluator.evaluate(
        dataset_name=request.dataset_name,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        rerank_top_k=request.rerank_top_k,
    )

    return EvalResponse(
        dataset=result.get("dataset", request.dataset_name),
        num_queries=result.get("num_queries", 0),
        metrics=result.get("metrics", {}),
        query_details=result.get("query_details", []),
        config_used={
            "retrieval_mode": cfg.get("retrieval", {}).get("mode"),
            "reranker": cfg.get("reranker", {}).get("provider"),
            "chunking": cfg.get("chunking", {}).get("strategy"),
        },
    )