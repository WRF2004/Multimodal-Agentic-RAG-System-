"""
Conversation history management with compression and coreference resolution.
"""

import structlog
from typing import Optional
from collections import defaultdict

from app.core.interfaces import ConversationManagerInterface, Message, LLMInterface

logger = structlog.get_logger()


class ConversationManager(ConversationManagerInterface):
    """
    Manages multi-turn conversation history with:
    - Sliding window compression
    - LLM-based summarization
    - Coreference resolution for pronouns
    """

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_history_turns: int = 20,
        compression_strategy: str = "sliding_window",
        window_size: int = 10,
        enable_coreference: bool = True,
    ):
        self.llm = llm
        self.max_history_turns = max_history_turns
        self.compression_strategy = compression_strategy
        self.window_size = window_size
        self.enable_coreference = enable_coreference
        self._histories: dict[str, list[Message]] = defaultdict(list)

    async def add_message(self, session_id: str, message: Message) -> None:
        self._histories[session_id].append(message)

    async def get_history(self, session_id: str, max_turns: int = None) -> list[Message]:
        max_turns = max_turns or self.max_history_turns
        history = self._histories.get(session_id, [])
        if len(history) > max_turns * 2:
            return history[-(max_turns * 2):]
        return history

    async def compress_history(self, session_id: str) -> list[Message]:
        history = self._histories.get(session_id, [])
        if len(history) <= self.window_size * 2:
            return history

        if self.compression_strategy == "summarization" and self.llm:
            return await self._summarize_history(session_id, history)
        elif self.compression_strategy == "sliding_window":
            return self._sliding_window(history)
        else:
            return self._truncate(history)

    def _sliding_window(self, history: list[Message]) -> list[Message]:
        """Keep the last N turns, preserving turn boundaries."""
        keep = self.window_size * 2
        if len(history) <= keep:
            return history
        # Always keep system message if present
        result = []
        if history and history[0].role == "system":
            result.append(history[0])
            history = history[1:]
        return result + history[-keep:]

    def _truncate(self, history: list[Message]) -> list[Message]:
        return history[-(self.max_history_turns * 2):]

    async def _summarize_history(
        self, session_id: str, history: list[Message]
    ) -> list[Message]:
        """Summarize older history into a concise summary using LLM."""
        if not self.llm:
            return self._sliding_window(history)

        keep = self.window_size * 2
        to_summarize = history[:-keep]
        recent = history[-keep:]

        if not to_summarize:
            return history

        summary_prompt = Message(
            role="user",
            content=(
                "请将以下对话历史压缩为简洁的摘要，保留所有关键事实、用户偏好和重要上下文：\n\n"
                + "\n".join(f"{m.role}: {m.content}" for m in to_summarize)
                + "\n\n请输出简洁的中文摘要："
            )
        )
        summary_msg = await self.llm.generate([summary_prompt])
        compressed = [
            Message(role="system", content=f"[历史对话摘要] {summary_msg.content}")
        ] + recent

        self._histories[session_id] = compressed
        return compressed

    async def resolve_coreference(self, query: str, history: list[Message]) -> str:
        """
        Use LLM to resolve pronouns and references in the current query
        based on conversation history.
        """
        if not self.enable_coreference or not self.llm or not history:
            return query

        # Check if query likely contains references
        reference_indicators = [
            "它", "他", "她", "这个", "那个", "这些", "那些", "上面", "之前",
            "its", "it", "this", "that", "these", "those", "them", "they",
            "he", "she", "the above", "previous"
        ]
        has_reference = any(ind in query.lower() for ind in reference_indicators)
        if not has_reference:
            return query

        recent = history[-6:]  # last 3 turns
        context = "\n".join(f"{m.role}: {m.content}" for m in recent)

        resolve_prompt = [
            Message(
                role="system",
                content=(
                    "你是一个指代消解助手。根据对话历史，将用户最新问题中的代词和指代表达"
                    "替换为具体的实体或概念，使问题可以独立理解。"
                    "只输出改写后的问题，不要输出其他内容。如果不需要改写，原样输出。"
                )
            ),
            Message(
                role="user",
                content=f"对话历史:\n{context}\n\n用户最新问题: {query}\n\n改写后的问题:"
            )
        ]
        result = await self.llm.generate(resolve_prompt, temperature=0.0, max_tokens=256)
        resolved = result.content.strip()
        if resolved and len(resolved) > 2:
            logger.info("coreference_resolved", original=query, resolved=resolved)
            return resolved
        return query