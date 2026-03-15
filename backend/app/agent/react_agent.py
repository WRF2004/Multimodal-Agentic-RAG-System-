"""
ReAct (Reason + Act) Agent implementation.
Follows the Thought -> Action -> Observation loop.
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

REACT_SYSTEM_PROMPT = """你是一个智能助手，能够通过推理和使用工具来回答用户的问题。

你可以使用以下工具:
{tool_descriptions}

请使用以下格式进行推理和行动:

Thought: 分析当前情况，决定下一步做什么
Action: 工具名称
Action Input: {{"param1": "value1", "param2": "value2"}}
Observation: [工具返回的结果]

... (可以重复 Thought/Action/Observation 多次)

当你有足够的信息回答用户问题时，使用:
Thought: 我现在有足够的信息来回答了
Final Answer: [你的最终回答]

重要规则:
1. 每次只能调用一个工具
2. Action Input 必须是有效的 JSON
3. 如果工具返回错误，尝试换一种方式
4. 基于检索到的文档回答时，需要引用来源
5. 如果无法找到答案，诚实说明
"""


@register_component(ComponentRegistry.AGENT, "react")
class ReActAgent(AgentInterface):

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

    def _build_system_prompt(self) -> str:
        tool_descs = []
        for name, tool in self.tools.items():
            params = json.dumps(tool.parameters_schema, ensure_ascii=False, indent=2)
            tool_descs.append(f"- {name}: {tool.description}\n  参数: {params}")
        return REACT_SYSTEM_PROMPT.format(tool_descriptions="\n".join(tool_descs))

    async def run(
        self,
        query: str,
        history: list[Message],
        session_config: dict = None,
    ) -> AgentResponse:
        system_msg = Message(role="system", content=self._build_system_prompt())
        messages = [system_msg] + history + [Message(role="user", content=query)]
        steps = []

        for iteration in range(self.max_iterations):
            response = await self.llm.generate(messages, temperature=0.1)
            text = response.content

            # Check for Final Answer
            if "Final Answer:" in text:
                answer = text.split("Final Answer:")[-1].strip()
                return AgentResponse(
                    answer=answer,
                    steps=steps,
                    metadata={"iterations": iteration + 1}
                )

            # Parse Thought, Action, Action Input
            action = self._parse_action(text)
            if action is None:
                # No clear action, treat as final answer
                return AgentResponse(
                    answer=text,
                    steps=steps,
                    metadata={"iterations": iteration + 1, "parse_failed": True}
                )

            # Execute tool
            tool = self.tools.get(action.tool)
            if tool is None:
                observation = f"错误: 未找到工具 '{action.tool}'。可用工具: {list(self.tools.keys())}"
            else:
                try:
                    result = await tool.execute(**action.tool_input)
                    observation = result.output
                except Exception as e:
                    observation = f"工具执行错误: {str(e)}"

            step = AgentStep(action=action, observation=observation)
            steps.append(step)

            # Add to messages
            messages.append(Message(role="assistant", content=text))
            messages.append(Message(
                role="user",
                content=f"Observation: {observation}"
            ))

            logger.info(
                "react_step",
                iteration=iteration,
                tool=action.tool,
                obs_len=len(observation)
            )

        # Max iterations reached
        return AgentResponse(
            answer="抱歉，我在规定步骤内无法完成回答。请尝试简化问题。",
            steps=steps,
            metadata={"iterations": self.max_iterations, "max_reached": True}
        )

    async def run_stream(
        self,
        query: str,
        history: list[Message],
        session_config: dict = None,
    ) -> AsyncIterator[dict]:
        system_msg = Message(role="system", content=self._build_system_prompt())
        messages = [system_msg] + history + [Message(role="user", content=query)]
        steps = []

        for iteration in range(self.max_iterations):
            yield {"type": "thinking", "content": f"🤔 推理中 (步骤 {iteration + 1})..."}

            response = await self.llm.generate(messages, temperature=0.1)
            text = response.content

            if "Final Answer:" in text:
                answer = text.split("Final Answer:")[-1].strip()
                yield {"type": "answer", "content": answer}
                yield {"type": "done", "steps": len(steps)}
                return

            action = self._parse_action(text)
            if action is None:
                yield {"type": "answer", "content": text}
                yield {"type": "done", "steps": len(steps)}
                return

            yield {
                "type": "action",
                "tool": action.tool,
                "input": action.tool_input,
                "reasoning": action.reasoning,
            }

            tool = self.tools.get(action.tool)
            if tool:
                try:
                    result = await tool.execute(**action.tool_input)
                    observation = result.output
                except Exception as e:
                    observation = f"工具执行错误: {str(e)}"
            else:
                observation = f"未找到工具 '{action.tool}'"

            yield {"type": "observation", "content": observation[:500]}

            step = AgentStep(action=action, observation=observation)
            steps.append(step)
            messages.append(Message(role="assistant", content=text))
            messages.append(Message(role="user", content=f"Observation: {observation}"))

        yield {"type": "answer", "content": "达到最大推理步骤，无法完成回答。"}
        yield {"type": "done", "steps": len(steps)}

    def _parse_action(self, text: str) -> AgentAction | None:
        """Parse ReAct format output to extract action."""
        try:
            thought = ""
            action_name = ""
            action_input = {}

            lines = text.strip().split("\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("Thought:"):
                    thought = line[len("Thought:"):].strip()
                elif line.startswith("Action:"):
                    action_name = line[len("Action:"):].strip()
                elif line.startswith("Action Input:"):
                    # Collect all remaining lines as potential JSON
                    json_str = line[len("Action Input:"):].strip()
                    if not json_str:
                        remaining = "\n".join(l.strip() for l in lines[i+1:])
                        json_str = remaining
                    try:
                        action_input = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to extract JSON from the string
                        import re
                        
                        match = re.search(r'\{.*\}', json_str, re.DOTALL)
                        if match:
                            action_input = json.loads(match.group())
                        else:
                            action_input = {"query": json_str}
                    break  # Stop after Action Input

            if not action_name:
                return None

            return AgentAction(
                tool=action_name,
                tool_input=action_input,
                reasoning=thought,
                action_id=str(uuid.uuid4()),
            )
        except Exception as e:
            logger.warning("action_parse_error", error=str(e), text=text[:200])
            return None