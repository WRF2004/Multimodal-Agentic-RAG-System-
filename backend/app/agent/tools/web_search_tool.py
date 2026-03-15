"""Web search tool (uses DuckDuckGo or configurable API)."""

import structlog
from app.agent.tools.base import BaseTool, ToolResult

logger = structlog.get_logger()


class WebSearchTool(BaseTool):

    def __init__(self, max_results: int = 5, **kwargs):
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "在互联网上搜索最新信息。当知识库中没有相关内容或需要实时信息时使用。"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询"
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, **kwargs) -> ToolResult:
        try:
            import httpx
            # Use DuckDuckGo instant answer API as default
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1},
                    timeout=10,
                )
                data = resp.json()
                abstract = data.get("AbstractText", "")
                related = data.get("RelatedTopics", [])[:self.max_results]
                results = []
                if abstract:
                    results.append(f"摘要: {abstract}")
                for topic in related:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append(topic["Text"])
                if results:
                    return ToolResult(output="\n\n".join(results), success=True)
                return ToolResult(output="未找到相关搜索结果。", success=True)
        except Exception as e:
            logger.error("web_search_error", error=str(e))
            return ToolResult(output=f"搜索出错: {str(e)}", success=False)