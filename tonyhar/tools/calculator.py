"""数学计算工具。"""

from tonyhar.tooling import ToolPolicy, tool


@tool(policy=ToolPolicy(parallel_safe=True))
async def calculator(expression: str) -> str:
    """执行数学计算。

    Args:
        expression: 数学表达式，例如 2+3*4。
    """
    return str(eval(expression))  # 生产环境应替换为 AST 白名单解析器
