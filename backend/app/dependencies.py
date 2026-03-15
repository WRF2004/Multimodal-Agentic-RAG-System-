"""
Dependency injection and component factory.
Creates and wires together all system components based on configuration.
"""

import structlog
from typing import Optional

from app.config import config_manager, ConfigManager
from app.core.registry import registry, ComponentRegistry
from app.core.interfaces import (
    LLMInterface, EmbeddingInterface, VectorStoreInterface,
    RetrieverInterface, RerankerInterface, ChunkerInterface,
    OCRInterface, ASRInterface, AgentInterface,
)
from app.conversation.manager import ConversationManager
from app.evaluation.evaluator import RAGEvaluator
from app.agent.tools.retrieval_tool import RetrievalTool
from app.agent.tools.calculator_tool import CalculatorTool
from app.agent.tools.web_search_tool import WebSearchTool
from app.queue.broker import InMemoryBroker

logger = structlog.get_logger()

# Import all modules to trigger @register_component decorators
import app.llm.openai_compatible
import app.embedding.openai_embedding
import app.vectorstore.chroma_store
import app.vectorstore.qdrant_store
import app.chunking.fixed
import app.chunking.recursive
import app.chunking.semantic
import app.retrieval.dense
import app.retrieval.sparse
import app.retrieval.hybrid
import app.reranker.bge_reranker
import app.multimodal.ocr
import app.multimodal.asr
import app.multimodal.parser
import app.agent.react_agent
import app.agent.planner_agent
import app.agent.function_calling_agent
import app.evaluation.metrics


class ComponentFactory:
    """
    Factory that builds component instances from a config dict.
    Supports per-session dynamic reconfiguration.
    """

    def __init__(self):
        self._cache = {}

    def clear_cache(self):
        self._cache.clear()

    def build_llm(self, cfg: dict) -> LLMInterface:
        llm_cfg = cfg.get("llm", {})
        return registry.create(
            ComponentRegistry.LLM,
            llm_cfg.get("provider", "openai_compatible"),
            base_url=llm_cfg.get("base_url", ""),
            api_key=llm_cfg.get("api_key", ""),
            model=llm_cfg.get("model", "gpt-4o"),
            timeout=llm_cfg.get("timeout", 60),
        )

    def build_embedding(self, cfg: dict) -> EmbeddingInterface:
        emb_cfg = cfg.get("embedding", {})
        provider = emb_cfg.get("provider", "openai")
        if provider == "sentence_transformer":
            return registry.create(
                ComponentRegistry.EMBEDDING, provider,
                model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
                _dimensions=emb_cfg.get("dimensions", 384),
            )
        return registry.create(
            ComponentRegistry.EMBEDDING, provider,
            base_url=emb_cfg.get("base_url", ""),
            api_key=emb_cfg.get("api_key", ""),
            model=emb_cfg.get("model", "text-embedding-3-small"),
            _dimensions=emb_cfg.get("dimensions", 1536),
            batch_size=emb_cfg.get("batch_size", 64),
        )

    def build_vectorstore(self, cfg: dict, embedding_dims: int = 1536) -> VectorStoreInterface:
        vs_cfg = cfg.get("vectorstore", {})
        provider = vs_cfg.get("provider", "chroma")
        provider_cfg = vs_cfg.get(provider, {})
        return registry.create(
            ComponentRegistry.VECTORSTORE, provider,
            vector_size=embedding_dims,
            **provider_cfg,
        )

    def build_retriever(self, cfg: dict, embedding: EmbeddingInterface, vectorstore: VectorStoreInterface) -> RetrieverInterface:
        ret_cfg = cfg.get("retrieval", {})
        mode = ret_cfg.get("mode", "hybrid")

        dense = registry.create(
            ComponentRegistry.RETRIEVER, "dense",
            embedding=embedding,
            vectorstore=vectorstore,
        )

        if mode == "dense":
            return dense

        sparse_cfg = ret_cfg.get("sparse", {})
        sparse = registry.create(
            ComponentRegistry.RETRIEVER, "sparse",
            k1=sparse_cfg.get("k1", 1.5),
            b=sparse_cfg.get("b", 0.75),
        )

        if mode == "sparse":
            return sparse

        # Hybrid
        hybrid_cfg = ret_cfg.get("hybrid", {})
        return registry.create(
            ComponentRegistry.RETRIEVER, "hybrid",
            dense_retriever=dense,
            sparse_retriever=sparse,
            dense_weight=hybrid_cfg.get("dense_weight", 0.7),
            sparse_weight=hybrid_cfg.get("sparse_weight", 0.3),
        )

    def build_reranker(self, cfg: dict) -> Optional[RerankerInterface]:
        rr_cfg = cfg.get("reranker", {})
        if not rr_cfg.get("enabled", True):
            return registry.create(ComponentRegistry.RERANKER, "none")
        provider = rr_cfg.get("provider", "none")
        provider_cfg = rr_cfg.get(provider, {})
        return registry.create(
            ComponentRegistry.RERANKER, provider,
            **provider_cfg,
        )

    def build_chunker(self, cfg: dict, embedding: EmbeddingInterface = None) -> ChunkerInterface:
        ch_cfg = cfg.get("chunking", {})
        strategy = ch_cfg.get("strategy", "recursive")
        strategy_cfg = ch_cfg.get(strategy, {})
        if strategy == "semantic":
            strategy_cfg["embedding"] = embedding
        return registry.create(
            ComponentRegistry.CHUNKER, strategy,
            **strategy_cfg,
        )

    def build_agent(self, cfg: dict, llm: LLMInterface, retriever: RetrieverInterface, reranker: RerankerInterface = None) -> AgentInterface:
        agent_cfg = cfg.get("agent", {})
        strategy = agent_cfg.get("strategy", "react")

        # Build tools
        tools = []
        tool_names = agent_cfg.get("tools", ["retrieval"])
        if "retrieval" in tool_names:
            tools.append(RetrievalTool(retriever=retriever, reranker=reranker))
        if "calculator" in tool_names:
            tools.append(CalculatorTool())
        if "web_search" in tool_names:
            tools.append(WebSearchTool())

        return registry.create(
            ComponentRegistry.AGENT, strategy,
            llm=llm,
            tools=tools,
            max_iterations=agent_cfg.get("max_iterations", 10),
        )

    def build_conversation_manager(self, cfg: dict, llm: LLMInterface = None) -> ConversationManager:
        conv_cfg = cfg.get("conversation", {})
        comp_cfg = conv_cfg.get("compression", {})
        return ConversationManager(
            llm=llm,
            max_history_turns=conv_cfg.get("max_history_turns", 20),
            compression_strategy=comp_cfg.get("strategy", "sliding_window"),
            window_size=comp_cfg.get("window_size", 10),
            enable_coreference=conv_cfg.get("coreference_resolution", True),
        )

    def build_all(self, cfg: dict) -> dict:
        """Build all components from config and return as dict."""
        llm = self.build_llm(cfg)
        embedding = self.build_embedding(cfg)
        vectorstore = self.build_vectorstore(cfg, embedding.dimensions)
        retriever = self.build_retriever(cfg, embedding, vectorstore)
        reranker = self.build_reranker(cfg)
        chunker = self.build_chunker(cfg, embedding)
        agent = self.build_agent(cfg, llm, retriever, reranker)
        conversation_manager = self.build_conversation_manager(cfg, llm)
        evaluator = RAGEvaluator(
            retriever=retriever,
            reranker=reranker,
            metrics=cfg.get("evaluation", {}).get("metrics", []),
            datasets_dir=cfg.get("evaluation", {}).get("datasets_dir", "./data/eval_datasets"),
        )

        return {
            "llm": llm,
            "embedding": embedding,
            "vectorstore": vectorstore,
            "retriever": retriever,
            "reranker": reranker,
            "chunker": chunker,
            "agent": agent,
            "conversation_manager": conversation_manager,
            "evaluator": evaluator,
        }


# Global factory
factory = ComponentFactory()