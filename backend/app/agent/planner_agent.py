"""
Plan-and-Solve Agent implementation.
First creates a step-by-step plan, then executes each step sequentially.
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

PLANNER_SYSTEM_PROMPT = """你是一个智能规划助手。你的工作方式分为两个阶段：

**阶段一：制定计划**
分析用户的问题，制定一个分步骤的执行计划。以 JSON 数组的形式输出计划：
```json
{{
  "plan": [
    {{"step": 1, "description": "步骤描述", "tool": "工具名称", "tool_input": {{"param": "value"}}}},
    {{"step": 2, "description": "步骤描述", "tool": "工具名称", "tool_input": {{"param": "value"}}}}
  ]
}}

**阶段二：执行并综合**
根据每一步的执行结果，综合所有信息给出最终回答。

可用工具:
{tool_descriptions}

重要：

- 计划要尽量简洁，不要超过5步
- 每个步骤必须指定一个可用的工具
- 如果问题简单，可以只有1-2步
  """

SYNTHESIZE_PROMPT = """根据以下执行结果，综合回答用户的问题。

用户问题: {query}

执行结果:
{results}

请给出完整、准确的回答，引用相关来源。"""

@register_component(ComponentRegistry.AGENT, "planner")
class PlannerAgent(AgentInterface):
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
        return PLANNER_SYSTEM_PROMPT.format(tool_descriptions="\n".join(tool_descs))

    async def _create_plan(self, query: str, history: list[Message]) -> list[dict]:
        """Use LLM to create an execution plan."""
        messages = [
            Message(role="system", content=self._build_system_prompt()),
            *history,
            Message(role="user", content=f"请为以下问题制定执行计划：\n{query}")
        ]
        response = await self.llm.generate(messages, temperature=0.1)
        text = response.content

        # Extract JSON plan
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                plan_data = json.loads(json_match.group())
                return plan_data.get("plan", [])
            except json.JSONDecodeError:
                pass

        # Fallback: single retrieval step
        return [{"step": 1, "description": "检索相关信息", "tool": "knowledge_retrieval", "tool_input": {"query": query}}]

    async def run(
        self,
        query: str,
        history: list[Message],
        session_config: dict = None,
    ) -> AgentResponse:
        # Phase 1: Plan
        plan = await self._create_plan(query, history)
        steps = []
        results = []

        # Phase 2: Execute
        for plan_step in plan[:self.max_iterations]:
            tool_name = plan_step.get("tool", "")
            tool_input = plan_step.get("tool_input", {})
            description = plan_step.get("description", "")

            tool = self.tools.get(tool_name)
            if tool is None:
                observation = f"工具 '{tool_name}' 不可用，跳过此步骤"
            else:
                try:
                    result = await tool.execute(**tool_input)
                    observation = result.output
                except Exception as e:
                    observation = f"执行错误: {str(e)}"

            action = AgentAction(
                tool=tool_name,
                tool_input=tool_input,
                reasoning=description,
                action_id=str(uuid.uuid4()),
            )
            step = AgentStep(action=action, observation=observation)
            steps.append(step)
            results.append(f"步骤 {plan_step.get('step', '?')}: {description}\n结果: {observation}")

        # Phase 3: Synthesize
        synth_prompt = SYNTHESIZE_PROMPT.format(
            query=query,
            results="\n\n---\n\n".join(results)
        )
        synth_messages = [
            Message(role="system", content="你是一个知识丰富的助手，请基于提供的信息回答问题。"),
            *history,
            Message(role="user", content=synth_prompt)
        ]
        final_response = await self.llm.generate(synth_messages, temperature=0.3)

        return AgentResponse(
            answer=final_response.content,
            steps=steps,
            metadata={"plan_steps": len(plan), "strategy": "planner"}
        )

    async def run_stream(
        self,
        query: str,
        history: list[Message],
        session_config: dict = None,
    ) -> AsyncIterator[dict]:
        yield {"type": "thinking", "content": "📋 正在制定执行计划..."}

        plan = await self._create_plan(query, history)
        yield {"type": "plan", "content": json.dumps(plan, ensure_ascii=False, indent=2)}

        steps = []
        results = []

        for plan_step in plan[:self.max_iterations]:
            tool_name = plan_step.get("tool", "")
            tool_input = plan_step.get("tool_input", {})
            description = plan_step.get("description", "")
            step_num = plan_step.get("step", "?")

            yield {
                "type": "action",
                "tool": tool_name,
                "input": tool_input,
                "reasoning": f"步骤 {step_num}: {description}",
            }

            tool = self.tools.get(tool_name)
            if tool:
                try:
                    result = await tool.execute(**tool_input)
                    observation = result.output
                except Exception as e:
                    observation = f"执行错误: {str(e)}"
            else:
                observation = f"工具 '{tool_name}' 不可用"

            yield {"type": "observation", "content": observation[:500]}

            action = AgentAction(tool=tool_name, tool_input=tool_input, reasoning=description)
            steps.append(AgentStep(action=action, observation=observation))
            results.append(f"步骤 {step_num}: {description}\n结果: {observation}")

        yield {"type": "thinking", "content": "🔄 正在综合所有结果..."}

        synth_prompt = SYNTHESIZE_PROMPT.format(
            query=query,
            results="\n\n---\n\n".join(results)
        )
        synth_messages = [
            Message(role="system", content="你是一个知识丰富的助手，请基于提供的信息回答问题。"),
            *history,
            Message(role="user", content=synth_prompt)
        ]

        async for token in self.llm.generate_stream(synth_messages, temperature=0.3):
            yield {"type": "token", "content": token}

        yield {"type": "done", "steps": len(steps)}