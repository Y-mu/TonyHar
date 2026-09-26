import json
import unittest

from tonyhar.resilience import Deadline
from tonyhar.tooling import (
    BaseTool,
    FunctionTool,
    ToolManager,
    ToolPolicy,
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
        "additionalProperties": False,
    }

    async def execute(self, text: str):
        return {"text": text}


class BrokenTool(BaseTool):
    name = "broken"
    description = "始终失败"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    async def execute(self):
        raise RuntimeError("测试异常")


@tool(
    name="decorated_search",
    description="搜索测试数据",
)
class DecoratedSearchTool(BaseTool):
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词。",
            },
            "limit": {
                "type": "integer",
                "description": "最大结果数量。",
                "default": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def execute(self, query: str, limit: int = 10) -> str:
        return f"{query}:{limit}"


@tool
async def decorated_function(query: str, limit: int = 10) -> str:
    """搜索函数工具。

    Args:
        query: 搜索关键词。
        limit: 最大结果数量。
    """
    return f"{query}:{limit}"


class ToolManagerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = ToolManager([EchoTool, BrokenTool])

    async def test_execute_returns_tool_result(self):
        request = ToolRequest(
            name="echo",
            arguments={"text": "你好"},
            tool_call_id="call-1",
        )

        result = await self.manager.execute(
            request,
            deadline=Deadline.after(1),
        )

        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_call_id, "call-1")
        self.assertEqual(result.data, {"text": "你好"})
        self.assertEqual(json.loads(result.content), {"text": "你好"})

    async def test_manager_builds_all_tool_instances_and_schemas(self):
        self.assertIsInstance(self.manager.get("echo"), EchoTool)
        self.assertEqual(len(self.manager.schemas()), 2)
        self.assertEqual(
            self.manager.schemas()[0]["function"]["name"],
            "echo",
        )

    async def test_tool_decorator_declares_class_schema(self):
        manager = ToolManager([DecoratedSearchTool])
        decorated_search = manager.get("decorated_search")

        self.assertIsInstance(decorated_search, DecoratedSearchTool)
        self.assertEqual(decorated_search.name, "decorated_search")
        self.assertEqual(
            decorated_search.parameters["properties"]["query"],
            {"type": "string", "description": "搜索关键词。"},
        )
        self.assertEqual(
            await decorated_search.execute(query="发动机"),
            "发动机:10",
        )

    async def test_tool_decorator_converts_function_to_base_tool(self):
        self.assertIsInstance(decorated_function, FunctionTool)
        manager = ToolManager([decorated_function])
        schema = manager.schemas()[0]["function"]
        self.assertEqual(schema["name"], "decorated_function")
        self.assertEqual(
            schema["parameters"]["properties"]["query"],
            {"type": "string", "description": "搜索关键词。"},
        )
        self.assertEqual(
            await manager.get("decorated_function").execute(query="发动机"),
            "发动机:10",
        )

    async def test_missing_tool_returns_failed_result(self):
        result = await self.manager.execute(
            ToolRequest(name="missing", tool_call_id="call-2"),
            deadline=Deadline.after(1),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_not_found")
        self.assertEqual(result.tool_call_id, "call-2")

    async def test_tool_exception_returns_failed_result(self):
        result = await self.manager.execute(
            ToolRequest(name="broken"),
            deadline=Deadline.after(1),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_execution_error")
        self.assertEqual(result.error_message, "测试异常")

    async def test_manager_rejects_raw_dict_request(self):
        with self.assertRaisesRegex(TypeError, "ToolRequest"):
            await self.manager.execute(  # type: ignore[arg-type]
                {"name": "echo", "arguments": {}},
                deadline=Deadline.after(1),
            )

    async def test_decorator_rejects_missing_function_type_hints(self):
        with self.assertRaisesRegex(TypeError, "必须提供类型提示"):

            @tool()
            async def invalid_tool(value) -> str:
                """缺少参数类型提示。"""
                return value

    async def test_manager_rejects_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "工具名称重复"):
            ToolManager([EchoTool, EchoTool])

    async def test_non_idempotent_tool_cannot_enable_retry(self):
        with self.assertRaisesRegex(ValueError, "幂等工具"):
            ToolPolicy(max_attempts=2, idempotent=False)


if __name__ == "__main__":
    unittest.main()
