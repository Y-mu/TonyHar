from core.tool import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "执行数学计算，输入表达式如 '2+3*4'"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"}
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> str:
        return str(eval(expression))  # 生产环境别用 eval