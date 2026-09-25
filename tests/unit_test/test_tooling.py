import json
import unittest

from tonyhar.resilience import Deadline
from tonyhar.tooling import (
    BaseTool,
    FunctionTool,
    ToolExecutor,
    ToolPolicy,
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

    async def execute(self, text: str):
        return {"text": text}


class BrokenTool(BaseTool):
    name = "broken"
    description = "始终失败"
    parameters = {"type": "object", "properties": {}}

    async def execute(self):
        raise RuntimeError("测试异常")


@tool
async def decorated_search(query: str, limit: int = 10) -> str:
    """搜索测试数据。

    Args:
        query: 搜索关键词。
        limit: 最大结果数量。
    """
    return f"{query}:{limit}"


class ToolExecutorTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(EchoTool())
        self.registry.register(BrokenTool())
        self.executor = ToolExecutor(self.registry)

    async def test_execute_returns_tool_result(self):
        request = ToolRequest(
            name="echo",
            arguments={"text": "你好"},
            tool_call_id="call-1",
        )

        result = await self.executor.execute(
            request,
            deadline=Deadline.after(1),
        )

        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_call_id, "call-1")
        self.assertEqual(result.data, {"text": "你好"})
        self.assertEqual(json.loads(result.content), {"text": "你好"})

    async def test_register_many_registers_all_tools(self):
        registry = ToolRegistry()

        registry.register_many(EchoTool(), BrokenTool())

        self.assertIsInstance(registry.get("echo"), EchoTool)
        self.assertIsInstance(registry.get("broken"), BrokenTool)
        self.assertEqual(len(registry.schemas()), 2)

    async def test_tool_decorator_builds_schema_from_signature(self):
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
        self.assertEqual(
            await decorated_search.execute(query="发动机"),
            "发动机:10",
        )

    async def test_tool_decorator_requires_parameter_type_hints(self):
        with self.assertRaisesRegex(TypeError, "必须提供类型提示"):

            @tool
            async def invalid_tool(value):
                """缺少参数类型提示。"""
                return value

    async def test_missing_tool_returns_failed_result(self):
        result = await self.executor.execute(
            ToolRequest(name="missing", tool_call_id="call-2"),
            deadline=Deadline.after(1),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_not_found")
        self.assertEqual(result.tool_call_id, "call-2")

    async def test_tool_exception_returns_failed_result(self):
        result = await self.executor.execute(
            ToolRequest(name="broken"),
            deadline=Deadline.after(1),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_execution_error")
        self.assertEqual(result.error_message, "测试异常")

    async def test_executor_rejects_raw_dict_request(self):
        with self.assertRaisesRegex(TypeError, "ToolRequest"):
            await self.executor.execute(  # type: ignore[arg-type]
                {"name": "echo", "arguments": {}},
                deadline=Deadline.after(1),
            )

    async def test_tool_decorator_rejects_sync_function(self):
        with self.assertRaisesRegex(TypeError, "async def"):

            @tool
            def sync_tool(value: str) -> str:
                """同步工具。"""
                return value

    async def test_non_idempotent_tool_cannot_enable_retry(self):
        with self.assertRaisesRegex(ValueError, "幂等工具"):
            ToolPolicy(max_attempts=2, idempotent=False)


if __name__ == "__main__":
    unittest.main()
