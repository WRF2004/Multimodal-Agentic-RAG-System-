"""Knowledge base retrieval tool."""

import structlog
from typing import Optional

from app.agent.tools.base import BaseTool, ToolResult
from app.core.interfaces import RetrieverInterface, RerankerInterface

logger = structlog.get_logger()


class RetrievalTool(BaseTool):

    def __init__(
        self,
        retriever: RetrieverInterface,
        reranker: Optional[RerankerInterface] = None,
        top_k: int = 5,
        rerank_top_k: int = 3,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    @property
    def name(self) -> str:
        return "knowledge_retrieval"

    @property
    def description(self) -> str:
        return (
            "从知识库中检索与查询相关的文档片段。"
            "当需要查找特定事实、数据或背景信息时使用此工具。"
            "输入应为具体、清晰的搜索查询。"
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的文档数量",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, top_k: int = None, **kwargs) -> ToolResult:
        try:
            top_k = top_k or self.top_k
            result = await self.retriever.retrieve(query, top_k=top_k)
            documents = result.documents

            if self.reranker and documents:
                documents = await self.reranker.rerank(
                    query, documents, top_k=self.rerank_top_k
                )

            if not documents:
                return ToolResult(
                    output="未找到相关文档。",
                    success=True,
                    metadata={"doc_count": 0}
                )

            formatted = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", "未知来源")
                score = f"{doc.score:.3f}" if doc.score else "N/A"
                formatted.append(
                    f"[文档 {i+1}] (来源: {source}, 相关度: {score})\n{doc.content}"
                )

            return ToolResult(
                output="\n\n---\n\n".join(formatted),
                success=True,
                metadata={
                    "doc_count": len(documents),
                    "sources": [d.metadata.get("source", "") for d in documents],
                }
            )
        except Exception as e:
            logger.error("retrieval_tool_error", error=str(e))
            return ToolResult(output=f"检索出错: {str(e)}", success=False)