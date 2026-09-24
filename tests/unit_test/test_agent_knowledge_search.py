import json
import unittest

from core.agent import Agent
from core.intent_planner import ExecutionMode, IntentPlan, IntentPlanner
from core.llm import BaseLLM, LLMResponse
from core.userIntentrecognizer import IntentResult
from tooling import BaseTool, ToolRegistry, ToolRequest


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

    def run(self, query, top_k=5):
        self.queries.append((query, top_k))
        return {"matches": [{"text": "检索证据"}]}


class RecordingListTool(BaseTool):
    name = "knowledge_list"
    description = "test list"
    parameters = {"type": "object", "properties": {}}

    def run(self):
        return {"documents": ["manual.txt"]}


class RecordingLLM(BaseLLM):
    def __init__(self):
        self.messages = None

    async def chat(self, messages, tools):
        self.messages = list(messages)
        return LLMResponse(content="根据检索证据作答")


class UnexpectedLLM(BaseLLM):
    async def chat(self, messages, tools):
        raise AssertionError("确定性路由不应继续调用 LLM")


class ThinkingToolLLM(BaseLLM):
    def __init__(self):
        self.calls = []

    async def chat(self, messages, tools):
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
    async def test_intent_route_returns_tool_request_without_executing_it(self):
        registry = ToolRegistry()
        registry.register(RecordingListTool())
        agent = Agent(
            llm=UnexpectedLLM(),
            tools=registry,
            intent_planner=IntentPlanner(KnowledgeListRecognizer()),
        )

        direct_answer = agent.intent_planner.plan("列出文档")

        self.assertIsInstance(direct_answer, IntentPlan)
        self.assertIs(direct_answer.mode, ExecutionMode.DIRECT_TOOL)
        self.assertEqual(len(direct_answer.tool_calls), 1)
        self.assertIsInstance(direct_answer.tool_calls[0], ToolRequest)
        self.assertEqual(direct_answer.tool_calls[0].name, "knowledge_list")

    async def test_knowledge_query_searches_before_llm_generation(self):
        registry = ToolRegistry()
        search = RecordingSearchTool()
        registry.register(search)
        llm = RecordingLLM()
        agent = Agent(
            llm=llm,
            tools=registry,
            intent_planner=IntentPlanner(KnowledgeQueryRecognizer()),
        )

        answer = await agent.invoke("文档中的维修建议是什么？")

        self.assertEqual(answer, "根据检索证据作答")
        self.assertEqual(search.queries, [("文档中的维修建议是什么？", 5)])
        self.assertEqual(llm.messages[-2]["role"], "assistant")
        self.assertEqual(llm.messages[-1]["role"], "tool")
        result = json.loads(llm.messages[-1]["content"])
        self.assertEqual(result["matches"][0]["text"], "检索证据")

    async def test_knowledge_list_route_returns_without_calling_llm(self):
        registry = ToolRegistry()
        registry.register(RecordingListTool())
        agent = Agent(
            llm=UnexpectedLLM(),
            tools=registry,
            intent_planner=IntentPlanner(KnowledgeListRecognizer()),
        )

        answer = await agent.invoke("列出文档")

        self.assertEqual(json.loads(answer), {"documents": ["manual.txt"]})

    async def test_reasoning_content_is_preserved_for_tool_follow_up(self):
        registry = ToolRegistry()
        registry.register(RecordingListTool())
        llm = ThinkingToolLLM()
        agent = Agent(
            llm=llm,
            tools=registry,
            intent_planner=IntentPlanner(ChatRecognizer()),
        )

        answer = await agent.invoke("请调用工具")

        self.assertEqual(answer, "工具调用完成")
        assistant_message = llm.calls[1][-2]
        self.assertEqual(
            assistant_message["reasoning_content"],
            "需要调用知识列表工具",
        )
        self.assertEqual(assistant_message["tool_calls"][0]["id"], "call-1")


if __name__ == "__main__":
    unittest.main()
