"""
Component Registry - Central hub for managing pluggable components.
Supports dynamic registration and retrieval of implementations.
"""

import structlog
from typing import Any, Type
from app.core.interfaces import (
    LLMInterface, EmbeddingInterface, VectorStoreInterface,
    RetrieverInterface, RerankerInterface, ChunkerInterface,
    DocumentParserInterface, OCRInterface, ASRInterface,
    AgentInterface, EvaluationMetricInterface,
)

logger = structlog.get_logger()


class ComponentRegistry:
    """
    Singleton registry for all pluggable components.
    Allows dynamic registration and instantiation.
    """

    _instance = None
    _registry: dict[str, dict[str, Type]] = {}
    _instances: dict[str, dict[str, Any]] = {}

    # Category constants
    LLM = "llm"
    EMBEDDING = "embedding"
    VECTORSTORE = "vectorstore"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    CHUNKER = "chunker"
    PARSER = "parser"
    OCR = "ocr"
    ASR = "asr"
    AGENT = "agent"
    METRIC = "metric"

    INTERFACE_MAP = {
        LLM: LLMInterface,
        EMBEDDING: EmbeddingInterface,
        VECTORSTORE: VectorStoreInterface,
        RETRIEVER: RetrieverInterface,
        RERANKER: RerankerInterface,
        CHUNKER: ChunkerInterface,
        PARSER: DocumentParserInterface,
        OCR: OCRInterface,
        ASR: ASRInterface,
        AGENT: AgentInterface,
        METRIC: EvaluationMetricInterface,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, category: str, name: str, cls_type: Type) -> None:
        """Register a component class under a category."""
        if category not in self._registry:
            self._registry[category] = {}
        interface = self.INTERFACE_MAP.get(category)
        if interface and not issubclass(cls_type, interface):
            raise TypeError(
                f"{cls_type.__name__} does not implement {interface.__name__}"
            )
        self._registry[category][name] = cls_type
        logger.info("component_registered", category=category, name=name)

    def get_class(self, category: str, name: str) -> Type:
        """Get a registered component class."""
        if category not in self._registry or name not in self._registry[category]:
            available = list(self._registry.get(category, {}).keys())
            raise KeyError(
                f"Component '{name}' not found in category '{category}'. "
                f"Available: {available}"
            )
        return self._registry[category][name]

    def create(self, category: str, name: str, **kwargs) -> Any:
        """Create (or retrieve cached) an instance of a component."""
        cache_key = f"{category}:{name}:{hash(frozenset(kwargs.items()) if kwargs else frozenset())}"
        if cache_key in self._instances.get(category, {}):
            return self._instances[category][cache_key]

        cls_type = self.get_class(category, name)
        instance = cls_type(**kwargs)

        if category not in self._instances:
            self._instances[category] = {}
        self._instances[category][cache_key] = instance
        return instance

    def clear_cache(self, category: str = None):
        """Clear cached instances."""
        if category:
            self._instances.pop(category, None)
        else:
            self._instances.clear()

    def list_components(self, category: str = None) -> dict:
        """List all registered components."""
        if category:
            return {category: list(self._registry.get(category, {}).keys())}
        return {cat: list(names.keys()) for cat, names in self._registry.items()}


# Global registry
registry = ComponentRegistry()


def register_component(category: str, name: str):
    """Decorator for registering components."""
    def decorator(cls):
        registry.register(category, name, cls)
        return cls
    return decorator