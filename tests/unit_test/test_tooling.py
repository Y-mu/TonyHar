import json
import unittest

from tooling import (
    BaseTool,
    FunctionTool,
    ToolExecutor,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    tool,
)


class EchoTool(BaseTool):
    name = "echo"
    description = "返回输入"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, text: str):
        return {"text": text}


class BrokenTool(BaseTool):
    name = "broken"
    description = "始终失败"
    parameters = {"type": "object", "properties": {}}

    def run(self):
        raise RuntimeError("测试异常")


@tool
def decorated_search(query: str, limit: int = 10) -> str:
    """搜索测试数据。

    Args:
        query: 搜索关键词。
        limit: 最大结果数量。
    """
    return f"{query}:{limit}"


class ToolExecutorTest(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(EchoTool())
        self.registry.register(BrokenTool())

    def test_invoke_returns_tool_result(self):
        request = ToolRequest(
            name="echo",
            arguments={"text": "你好"},
            tool_call_id="call-1",
        )

        result = ToolExecutor.invoke(request, self.registry)

        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_call_id, "call-1")
        self.assertEqual(result.data, {"text": "你好"})
        self.assertEqual(json.loads(result.content), {"text": "你好"})

    def test_register_many_registers_all_tools(self):
        registry = ToolRegistry()

        registry.register_many(EchoTool(), BrokenTool())

        self.assertIsInstance(registry.get("echo"), EchoTool)
        self.assertIsInstance(registry.get("broken"), BrokenTool)
        self.assertEqual(len(registry.schemas()), 2)

    def test_tool_decorator_builds_schema_from_signature(self):
        self.assertIsInstance(decorated_search, FunctionTool)
        self.assertEqual(decorated_search.name, "decorated_search")
        self.assertIn("搜索测试数据", decorated_search.description)
        self.assertEqual(
            decorated_search.parameters["properties"]["query"],
            {"type": "string", "description": "搜索关键词。"},
        )
        self.assertEqual(
            decorated_search.parameters["properties"]["limit"],
            {
                "type": "integer",
                "description": "最大结果数量。",
                "default": 10,
            },
        )
        self.assertEqual(decorated_search.parameters["required"], ["query"])
        self.assertEqual(decorated_search(query="发动机"), "发动机:10")

    def test_tool_decorator_requires_parameter_type_hints(self):
        with self.assertRaisesRegex(TypeError, "必须提供类型提示"):

            @tool
            def invalid_tool(value):
                """缺少参数类型提示。"""
                return value

    def test_missing_tool_returns_failed_result(self):
        result = ToolExecutor.invoke(
            ToolRequest(name="missing", tool_call_id="call-2"),
            self.registry,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_not_found")
        self.assertEqual(result.tool_call_id, "call-2")

    def test_tool_exception_returns_failed_result(self):
        result = ToolExecutor.invoke(
            ToolRequest(name="broken"),
            self.registry,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_execution_error")
        self.assertEqual(result.error_message, "测试异常")

    def test_executor_rejects_raw_dict_request(self):
        with self.assertRaisesRegex(TypeError, "ToolRequest"):
            ToolExecutor.invoke(  # type: ignore[arg-type]
                {"name": "echo", "arguments": {}},
                self.registry,
            )


if __name__ == "__main__":
    unittest.main()
