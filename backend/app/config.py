"""
Centralized configuration management.
Loads from YAML + environment variables with runtime override support.
"""

import os
import yaml
import structlog
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from copy import deepcopy

logger = structlog.get_logger()


class LLMConfig(BaseModel):
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64


class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    qdrant: dict = Field(default_factory=lambda: {
        "url": "http://localhost:6333",
        "collection_name": "documents"
    })
    chroma: dict = Field(default_factory=lambda: {
        "persist_directory": "./data/chroma",
        "collection_name": "documents"
    })
    pgvector: dict = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    mode: str = "hybrid"
    top_k: int = 20
    hybrid: dict = Field(default_factory=lambda: {
        "dense_weight": 0.7,
        "sparse_weight": 0.3
    })
    sparse: dict = Field(default_factory=lambda: {
        "algorithm": "bm25", "k1": 1.5, "b": 0.75
    })


class RerankerConfig(BaseModel):
    enabled: bool = True
    provider: str = "bge"
    model: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5
    bge: dict = Field(default_factory=dict)
    cohere: dict = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    strategy: str = "semantic"
    fixed: dict = Field(default_factory=lambda: {
        "chunk_size": 512, "chunk_overlap": 50
    })
    recursive: dict = Field(default_factory=lambda: {
        "chunk_size": 512, "chunk_overlap": 50,
        "separators": ["\n\n", "\n", "。", ".", " "]
    })
    semantic: dict = Field(default_factory=lambda: {
        "max_chunk_size": 1024, "min_chunk_size": 100,
        "similarity_threshold": 0.75
    })


class AgentConfig(BaseModel):
    strategy: str = "react"
    max_iterations: int = 10
    tools: list[str] = Field(default_factory=lambda: ["retrieval", "web_search", "calculator"])


class MultimodalConfig(BaseModel):
    ocr: dict = Field(default_factory=lambda: {
        "enabled": True, "provider": "tesseract"
    })
    asr: dict = Field(default_factory=lambda: {
        "enabled": True, "provider": "whisper",
        "whisper": {"model_size": "base"}
    })


class ConversationConfig(BaseModel):
    max_history_turns: int = 20
    compression: dict = Field(default_factory=lambda: {
        "strategy": "sliding_window", "window_size": 10
    })
    coreference_resolution: bool = True


class DatabaseConfig(BaseModel):
    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_rag"
    echo: bool = False
    pool_size: int = 20


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    ttl: int = 3600


class RabbitMQConfig(BaseModel):
    url: str = "amqp://guest:guest@localhost:5672/"
    queue_name: str = "document_processing"


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: ["recall", "mrr", "ndcg", "precision", "hit_rate"]
    )
    datasets_dir: str = "./data/eval_datasets"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )


class SystemConfig(BaseModel):
    """Top-level system configuration."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    rabbitmq: RabbitMQConfig = Field(default_factory=RabbitMQConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} in config values."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, "")
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


class ConfigManager:
    """
    Manages system configuration with support for:
    - YAML file loading
    - Environment variable resolution
    - Per-session runtime overrides
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._base_config = None
            cls._instance._session_overrides: dict[str, dict] = {}
        return cls._instance

    def load(self, config_path: str = None) -> SystemConfig:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.environ.get(
                "CONFIG_PATH",
                str(Path(__file__).parent.parent / "config" / "default.yaml")
            )

        raw = {}
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                raw = yaml.safe_load(f) or {}
            logger.info("config_loaded", path=config_path)
        else:
            logger.warning("config_file_not_found", path=config_path)

        resolved = _resolve_env_vars(raw)
        self._base_config = SystemConfig(**resolved)
        return self._base_config

    @property
    def config(self) -> SystemConfig:
        if self._base_config is None:
            return self.load()
        return self._base_config

    def get_session_config(self, session_id: str) -> dict:
        """Get merged config for a specific session (base + overrides)."""
        base = self.config.model_dump()
        overrides = self._session_overrides.get(session_id, {})
        return self._deep_merge(base, overrides)

    def set_session_override(self, session_id: str, overrides: dict) -> None:
        """Set runtime configuration overrides for a session."""
        self._session_overrides[session_id] = overrides
        logger.info("session_config_updated", session_id=session_id, keys=list(overrides.keys()))

    def clear_session_override(self, session_id: str) -> None:
        self._session_overrides.pop(session_id, None)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result


config_manager = ConfigManager()