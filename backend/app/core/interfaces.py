"""
Core abstract interfaces for all pluggable components.
Every major module implements one of these interfaces, enabling seamless swapping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional
from enum import Enum


# ============================================================
# Data Models
# ============================================================

@dataclass
class Document:
    """Represents a parsed document or chunk."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    score: Optional[float] = None
    source_type: str = "text"  # text | image | audio | video


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    metadata: dict = field(default_factory=dict)
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class RetrievalResult:
    documents: list[Document]
    query: str
    strategy_used: str
    scores: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentAction:
    tool: str
    tool_input: dict
    reasoning: str
    action_id: str = ""


@dataclass
class AgentStep:
    action: AgentAction
    observation: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[Document] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class AgentStrategy(str, Enum):
    REACT = "react"
    PLANNER = "planner"
    FUNCTION_CALLING = "function_calling"


# ============================================================
# Abstract Interfaces
# ============================================================

class LLMInterface(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Message:
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> AsyncIterator[str]:
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        **kwargs
    ) -> Message:
        ...


class EmbeddingInterface(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...


class VectorStoreInterface(ABC):
    """Abstract interface for vector databases."""

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        ...

    @abstractmethod
    async def search(
        self, query_embedding: list[float], top_k: int = 10, filters: Optional[dict] = None
    ) -> list[Document]:
        ...

    @abstractmethod
    async def delete(self, document_ids: list[str]) -> None:
        ...

    @abstractmethod
    async def get_collection_stats(self) -> dict:
        ...


class RetrieverInterface(ABC):
    """Abstract interface for retrievers."""

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[dict] = None
    ) -> RetrievalResult:
        ...


class RerankerInterface(ABC):
    """Abstract interface for rerankers."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 5
    ) -> list[Document]:
        ...


class ChunkerInterface(ABC):
    """Abstract interface for text chunking strategies."""

    @abstractmethod
    async def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        ...


class DocumentParserInterface(ABC):
    """Abstract interface for document parsers."""

    @abstractmethod
    async def parse(self, file_path: str, file_type: str) -> list[Document]:
        ...

    @abstractmethod
    def supported_types(self) -> list[str]:
        ...


class OCRInterface(ABC):
    """Abstract interface for OCR engines."""

    @abstractmethod
    async def extract_text(self, image_path: str) -> Optional[str]:
        ...


class ASRInterface(ABC):
    """Abstract interface for ASR engines."""

    @abstractmethod
    async def transcribe(self, audio_path: str) -> Optional[str]:
        ...


class AgentInterface(ABC):
    """Abstract interface for agent strategies."""

    @abstractmethod
    async def run(
        self,
        query: str,
        history: list[Message],
        session_config: dict,
    ) -> AgentResponse:
        ...

    @abstractmethod
    async def run_stream(
        self,
        query: str,
        history: list[Message],
        session_config: dict,
    ) -> AsyncIterator[dict]:
        ...


class ConversationManagerInterface(ABC):
    """Abstract interface for conversation history management."""

    @abstractmethod
    async def add_message(self, session_id: str, message: Message) -> None:
        ...

    @abstractmethod
    async def get_history(self, session_id: str, max_turns: int = 20) -> list[Message]:
        ...

    @abstractmethod
    async def compress_history(self, session_id: str) -> list[Message]:
        ...

    @abstractmethod
    async def resolve_coreference(self, query: str, history: list[Message]) -> str:
        ...


class EvaluationMetricInterface(ABC):
    """Abstract interface for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute(
        self,
        retrieved: list[list[str]],
        relevant: list[list[str]],
        **kwargs
    ) -> float:
        ...


class MessageBrokerInterface(ABC):
    """Abstract interface for message queue."""

    @abstractmethod
    async def publish(self, queue: str, message: dict) -> None:
        ...

    @abstractmethod
    async def consume(self, queue: str, callback) -> None:
        ...