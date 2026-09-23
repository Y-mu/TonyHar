import json
from uuid import uuid4

from .llm import BaseLLM
from .tool import ToolRegistry
from .memory import Memory
from .userIntentrecognizer import (
    UserIntentRecognizer,
    get_shared_intent_recognizer,
)

class Agent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        system_prompt: str = "",
        intent_recognizer: UserIntentRecognizer | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = Memory()
        if system_prompt:
            self.memory.add("system", system_prompt)
        self.max_steps = 10
        self.intent_recognizer = intent_recognizer or get_shared_intent_recognizer()

    async def run(self, user_input: str) -> str:
        original_input = user_input
        # 意图识别
        route = self.intent_recognizer.classify(user_input)
        if route.name == "knowledge_list":
            self.memory.add("user", user_input)
            answer = self.tools.execute("knowledge_list", {})
            self.memory.add("assistant", answer)
            return answer
        if route.name == "ingest":
            self.memory.add("user", user_input)
            answer = self.tools.execute(
                "file_ingestion",
                {"message": user_input},
            )
            self.memory.add("assistant", answer)
            return answer
        confidence = (
            f", confidence={route.score:.3f}"
            if route.score is not None else ""
        )

        user_input = f"{user_input}\n\n[语义路由: {route.name}{confidence}]"
        if route.name == "knowledge_query":
            user_input += "\n请优先考虑从已入库的知识内容回答。"
        self.memory.add("user", user_input)

        # Knowledge-base questions always retrieve evidence before generation.
        # Recording a normal assistant tool call keeps the message history valid
        # for OpenAI-compatible chat APIs and still allows follow-up tool calls.
        if route.name == "knowledge_query":
            call_id = f"forced_knowledge_search_{uuid4().hex}"
            arguments = {"query": original_input, "top_k": 5}
            self.memory.add(
                "assistant",
                "",
                tool_calls=[{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "knowledge_search",
                        "arguments": json.dumps(
                            arguments,
                            ensure_ascii=False,
                        ),
                    },
                }],
            )
            result = self.tools.execute("knowledge_search", arguments)
            self.memory.add("tool", result, tool_call_id=call_id)

        step_total = 0

        for step in range(self.max_steps):
            # 1. 调用 LLM
            response = await self.llm.chat(
                messages=self.memory.get(),
                tools=self.tools.schemas(),
            )

            # 2. 没有工具调用 → 最终答案
            if not response.get_tool_calls():
                answer = response.get_content()
                self.memory.add("assistant", answer)
                return answer

            # 3. 有工具调用 → 执行
            self.memory.add("assistant", 
                            response.get_content(), 
                            tool_calls=response.get_tool_calls_as_dicts())

            for call in response.get_tool_calls():
                name = call.name
                args = call.arguments
                result = self.tools.execute(name, args)

                # 4. 把结果作为 observation 加回
                self.memory.add("tool", result, tool_call_id=call.id)
            step_total = step
        print("Warning: Reached maximum steps without a final answer.")
        return step_total
