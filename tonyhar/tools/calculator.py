"""数学计算工具。"""

from typing import Any

from tonyhar.tooling import BaseTool, ToolPolicy, tool


@tool(policy=ToolPolicy(parallel_safe=True))
class CalculatorTool(BaseTool):
    """执行数学计算。"""

    name = "calculator"
    description = "执行数学计算。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如 2+3*4。",
            },
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    async def execute(self, expression: str) -> str:
        return str(eval(expression))  # 生产环境应替换为 AST 白名单解析器
