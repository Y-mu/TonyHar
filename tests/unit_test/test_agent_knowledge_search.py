import json
import unittest

from tonyhar.agent import Agent
from tonyhar.agent.intent_planner import ExecutionMode, IntentPlan, IntentPlanner
from tonyhar.agent.llm import BaseLLM, LLMResponse, ModelUnavailableError
from tonyhar.agent.runnables import AgentRunContext, AgentState
from tonyhar.agent.user_intent_recognizer import IntentResult
from tonyhar.resilience import Deadline, RunDeadlineExceeded
from tonyhar.tooling import BaseTool, ToolManager, ToolRequest
from tonyhar.tools.knowledge_list import format_knowledge_list


class KnowledgeQueryRecognizer:
    def classify(self, text):
        return IntentResult(name="knowledge_query", score=0.9)


class KnowledgeListRecognizer:
    def classify(self, text):
        return IntentResult(name="knowledge_list", score=0.9)


class ChatRecognizer:
    def classify(self, text):
        return IntentResult(name="chat", score=0.9)


class RecordingSearchTool(BaseTool):
    name = "knowledge_search"
    description = "test search"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.queries = []

    async def execute(self, query, top_k=5):
        self.queries.append((query, top_k))
        return {"matches": [{"text": "检索证据"}]}


class RecordingListTool(BaseTool):
    name = "knowledge_list"
    description = "test list"
    parameters = {"type": "object", "properties": {}}

    async def execute(self):
        return {"documents": ["manual.txt"]}

    def format_result(self, data):
        return format_knowledge_list(data)


class TestLLM(BaseLLM):
    async def aclose(self):
        return None


class RecordingLLM(TestLLM):
    def __init__(self):
        self.messages = None
        self.tools = None

    async def chat(self, messages, tools, *, deadline):
        self.messages = list(messages)
        self.tools = list(tools)
        return LLMResponse(content="根据检索证据作答")


class UnexpectedLLM(TestLLM):
    async def chat(self, messages, tools, *, deadline):
        raise AssertionError("确定性路由不应继续调用 LLM")


class FailingLLM(TestLLM):
    async def chat(self, messages, tools, *, deadline):
        raise ModelUnavailableError("model unavailable")


class DeadlineLLM(TestLLM):
    async def chat(self, messages, tools, *, deadline):
        raise RunDeadlineExceeded()


class ThinkingToolLLM(TestLLM):
    def __init__(self):
        self.calls = []

    async def chat(self, messages, tools, *, deadline):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return LLMResponse(
                reasoning_content="需要调用知识列表工具",
                tool_calls=[ToolRequest(
                    tool_call_id="call-1",
                    name="knowledge_list",
                )],
            )
        return LLMResponse(content="工具调用完成")


class AgentKnowledgeSearchTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def context(user_input: str) -> AgentRunContext:
        return AgentRunContext(
            run_id="run-test",
            session_id="session-test",
            user_input=user_input,
            messages=[],
            deadline=Deadline.after(1),
        )

    async def test_intent_route_returns_tool_request_without_executing_it(self):
        tool_manager = ToolManager([RecordingListTool])
        agent = Agent(
            llm=UnexpectedLLM(),
            tool_manager=tool_manager,
            intent_planner=IntentPlanner(KnowledgeListRecognizer()),
        )

        direct_answer = agent.intent_planner.plan("列出文档")

        self.assertIsInstance(direct_answer, IntentPlan)
        self.assertIs(direct_answer.mode, ExecutionMode.DIRECT_TOOL)
        self.assertEqual(len(direct_answer.tool_calls), 1)
        self.assertIsInstance(direct_answer.tool_calls[0], ToolRequest)
        self.assertEqual(direct_answer.tool_calls[0].name, "knowledge_list")

    async def test_knowledge_query_searches_before_llm_generation(self):
        search = RecordingSearchTool()
        tool_manager = ToolManager([lambda: search])
        llm = RecordingLLM()
        agent = Agent(
            llm=llm,
            tool_manager=tool_manager,
            intent_planner=IntentPlanner(KnowledgeQueryRecognizer()),
        )

        context = self.context("文档中的维修建议是什么？")
        result = await agent.invoke(context)

        self.assertTrue(result.success)
        self.assertEqual(result.answer, "根据检索证据作答")
        self.assertIs(context.state, AgentState.COMPLETED)
        self.assertEqual(search.queries, [("文档中的维修建议是什么？", 5)])
        self.assertEqual(
            [schema["function"]["name"] for schema in llm.tools],
            ["knowledge_search"],
        )
        self.assertEqual(llm.messages[-2]["role"], "assistant")
        self.assertEqual(llm.messages[-1]["role"], "tool")
        result = json.loads(llm.messages[-1]["content"])
        self.assertEqual(result["matches"][0]["text"], "检索证据")

    async def test_runtime_error_returns_structured_failed_result(self):
        agent = Agent(
            llm=FailingLLM(),
            tool_manager=ToolManager(),
            intent_planner=IntentPlanner(ChatRecognizer()),
        )
        context = self.context("你好")

        result = await agent.invoke(context)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "model_unavailable")
        self.assertEqual(result.error_message, "model unavailable")
        self.assertIs(context.state, AgentState.FAILED)
        self.assertFalse(hasattr(agent, "memory"))

    async def test_knowledge_list_route_returns_without_calling_llm(self):
        tool_manager = ToolManager([RecordingListTool])
        agent = Agent(
            llm=UnexpectedLLM(),
            tool_manager=tool_manager,
            intent_planner=IntentPlanner(KnowledgeListRecognizer()),
        )

        context = self.context("列出文档")
        result = await agent.invoke(context)

        self.assertTrue(result.success)
        self.assertEqual(
            result.answer,
            "## 知识库文档\n\n"
            "当前共有 **1** 篇已入库文档：\n\n"
            "1. `manual.txt`",
        )
        self.assertIs(context.state, AgentState.COMPLETED)

    async def test_run_deadline_has_stable_terminal_error_code(self):
        agent = Agent(
            llm=DeadlineLLM(),
            tool_manager=ToolManager(),
            intent_planner=IntentPlanner(ChatRecognizer()),
        )

        result = await agent.invoke(self.context("你好"))

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "timeout")
        self.assertEqual(result.error_code, "run_timeout")

    async def test_reasoning_content_is_preserved_for_tool_follow_up(self):
        tool_manager = ToolManager([RecordingListTool])
        llm = ThinkingToolLLM()
        agent = Agent(
            llm=llm,
            tool_manager=tool_manager,
            intent_planner=IntentPlanner(ChatRecognizer()),
        )

        context = self.context("请调用工具")
        result = await agent.invoke(context)

        self.assertTrue(result.success)
        self.assertEqual(result.answer, "工具调用完成")
        assistant_message = llm.calls[1][-2]
        self.assertEqual(
            assistant_message["reasoning_content"],
            "需要调用知识列表工具",
        )
        self.assertEqual(assistant_message["tool_calls"][0]["id"], "call-1")


if __name__ == "__main__":
    unittest.main()
