"""Calculator tool for mathematical computations."""

import ast
import operator
from app.agent.tools.base import BaseTool, ToolResult


class CalculatorTool(BaseTool):

    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "执行数学计算。输入数学表达式字符串，如 '2 + 3 * 4' 或 '(10 / 3) ** 2'。"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式"
                }
            },
            "required": ["expression"]
        }

    def _safe_eval(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant")
        elif isinstance(node, ast.BinOp):
            op = self.SAFE_OPERATORS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(self._safe_eval(node.left), self._safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = self.SAFE_OPERATORS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported unary: {type(node.op)}")
            return op(self._safe_eval(node.operand))
        raise ValueError(f"Unsupported node: {type(node)}")

    async def execute(self, expression: str, **kwargs) -> ToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._safe_eval(tree.body)
            return ToolResult(output=str(result), success=True)
        except Exception as e:
            return ToolResult(output=f"计算错误: {str(e)}", success=False)