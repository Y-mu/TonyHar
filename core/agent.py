import json
from .llm import BaseLLM
from .tool import ToolRegistry
from .memory import Memory
from .semantic_router import SemanticIntentRouter

class Agent:
    def __init__(self, llm: BaseLLM, tools: ToolRegistry, system_prompt: str = "", semantic_router=None):
        self.llm = llm
        self.tools = tools
        self.memory = Memory()
        if system_prompt:
            self.memory.add("system", system_prompt)
        self.max_steps = 10
        self.semantic_router = semantic_router or SemanticIntentRouter()

    async def run(self, user_input: str) -> str:
        # 路由属于 Agent 的输入处理层，不由 main.py 决定分支。
        route = self.semantic_router.classify(user_input)
        confidence = (
            f", confidence={route.score:.3f}"
            if route.score is not None else ""
        )
        user_input = f"{user_input}\n\n[语义路由: {route.name}{confidence}]"
        if route.name == "knowledge_query":
            user_input += "\n请优先考虑从已入库的知识内容回答。"
        self.memory.add("user", user_input)

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
