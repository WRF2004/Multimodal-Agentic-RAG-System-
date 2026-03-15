"""
Function Calling Agent implementation.
Uses native OpenAI function calling / tool_use API for structured tool invocation.
"""

import json
import uuid
import structlog
from typing import AsyncIterator

from app.core.interfaces import (
    AgentInterface, AgentResponse, AgentAction, AgentStep,
    Message, LLMInterface
)
from app.core.registry import register_component, ComponentRegistry
from app.agent.tools.base import BaseTool

logger = structlog.get_logger()

FC_SYSTEM_PROMPT = """你是一个智能助手，能够使用工具来回答用户的问题。
请根据用户的问题，选择合适的工具获取信息，然后基于获取的信息给出准确、完整的回答。
如果知识库中没有相关信息，请诚实告知。回答时请引用信息来源。"""


@register_component(ComponentRegistry.AGENT, "function_calling")
class FunctionCallingAgent(AgentInterface):

    def __init__(
        self,
        llm: LLMInterface,
        tools: list[BaseTool],
        max_iterations: int = 10,
        **kwargs
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def _get_openai_tools(self) -> list[dict]:
        return [tool.to_openai_tool() for tool in self.tools.values()]

    async def run(
        self,
        query: str,
        history: list[Message],
        session_config: dict = None,
    ) -> AgentResponse:
        messages = [
            Message(role="system", content=FC_SYSTEM_PROMPT),
            *history,
            Message(role="user", content=query),
        ]
        tools_schema = self._get_openai_tools()
        steps = []

        for iteration in range(self.max_iterations):
            response = await self.llm.generate_with_tools(messages, tools=tools_schema)

            # If no tool calls, return the response as final answer
            if not response.tool_calls:
                return AgentResponse(
                    answer=response.content,
                    steps=steps,
                    metadata={"iterations": iteration + 1, "strategy": "function_calling"}
                )

            # Process each tool call
            messages.append(response)

            for tool_call in response.tool_calls:
                func_name = tool_call["function"]["name"]
                try:
                    func_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                tool = self.tools.get(func_name)
                if tool is None:
                    observation = f"未找到工具: {func_name}"
                else:
                    try:
                        result = await tool.execute(**func_args)
                        observation = result.output
                    except Exception as e:
                        observation = f"工具执行错误: {str(e)}"

                action = AgentAction(
                    tool=func_name,
                    tool_input=func_args,
                    reasoning=f"Function call: {func_name}",
                    action_id=tool_call.get("id", str(uuid.uuid4())),
                )
                steps.append(AgentStep(action=action, observation=observation))

                # Add tool result as message
                messages.append(Message(
                    role="tool",
                    content=observation,
                    tool_call_id=tool_call.get("id", ""),
                ))

            logger.info("fc_iteration", iteration=iteration, tool_calls=len(response.tool_calls))

        return AgentResponse(
            answer="达到最大迭代次数，无法完成回答。",
            steps=steps,
            metadata={"iterations": self.max_iterations, "max_reached": True}
        )

    async def run_stream(
        self,
        query: str,
        history: list[Message],
        session_config: dict = None,
    ) -> AsyncIterator[dict]:
        messages = [
            Message(role="system", content=FC_SYSTEM_PROMPT),
            *history,
            Message(role="user", content=query),
        ]
        tools_schema = self._get_openai_tools()
        steps = []

        for iteration in range(self.max_iterations):
            yield {"type": "thinking", "content": f"🔧 函数调用中 (轮次 {iteration + 1})..."}

            response = await self.llm.generate_with_tools(messages, tools=tools_schema)

            if not response.tool_calls:
                # Stream the final answer
                if response.content:
                    yield {"type": "answer", "content": response.content}
                yield {"type": "done", "steps": len(steps)}
                return

            messages.append(response)

            for tool_call in response.tool_calls:
                func_name = tool_call["function"]["name"]
                try:
                    func_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                yield {
                    "type": "action",
                    "tool": func_name,
                    "input": func_args,
                    "reasoning": f"调用工具: {func_name}",
                }

                tool = self.tools.get(func_name)
                if tool:
                    try:
                        result = await tool.execute(**func_args)
                        observation = result.output
                    except Exception as e:
                        observation = f"工具执行错误: {str(e)}"
                else:
                    observation = f"未找到工具: {func_name}"

                yield {"type": "observation", "content": observation[:500]}

                action = AgentAction(
                    tool=func_name,
                    tool_input=func_args,
                    reasoning=f"Function call: {func_name}",
                    action_id=tool_call.get("id", ""),
                )
                steps.append(AgentStep(action=action, observation=observation))

                messages.append(Message(
                    role="tool",
                    content=observation,
                    tool_call_id=tool_call.get("id", ""),
                ))

        yield {"type": "answer", "content": "达到最大迭代次数，无法完成回答。"}
        yield {"type": "done", "steps": len(steps)}