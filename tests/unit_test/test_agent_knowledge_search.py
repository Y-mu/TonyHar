import json
import unittest

from core.agent import Agent
from core.llm import BaseLLM, LLMResponse
from core.tool import BaseTool, ToolRegistry
from core.userIntentrecognizer import IntentResult


class KnowledgeQueryRecognizer:
    def classify(self, text):
        return IntentResult(name="knowledge_query", score=0.9)


class RecordingSearchTool(BaseTool):
    name = "knowledge_search"
    description = "test search"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.queries = []

    def run(self, query, top_k=5):
        self.queries.append((query, top_k))
        return {"matches": [{"text": "检索证据"}]}


class RecordingLLM(BaseLLM):
    def __init__(self):
        self.messages = None

    async def chat(self, messages, tools):
        self.messages = list(messages)
        return LLMResponse(content="根据检索证据作答")


class AgentKnowledgeSearchTest(unittest.IsolatedAsyncioTestCase):
    async def test_knowledge_query_searches_before_llm_generation(self):
        registry = ToolRegistry()
        search = RecordingSearchTool()
        registry.register(search)
        llm = RecordingLLM()
        agent = Agent(
            llm=llm,
            tools=registry,
            intent_recognizer=KnowledgeQueryRecognizer(),
        )

        answer = await agent.run("文档中的维修建议是什么？")

        self.assertEqual(answer, "根据检索证据作答")
        self.assertEqual(search.queries, [("文档中的维修建议是什么？", 5)])
        self.assertEqual(llm.messages[-2]["role"], "assistant")
        self.assertEqual(llm.messages[-1]["role"], "tool")
        result = json.loads(llm.messages[-1]["content"])
        self.assertEqual(result["matches"][0]["text"], "检索证据")


if __name__ == "__main__":
    unittest.main()
