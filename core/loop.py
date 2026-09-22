
from llm import LLMClient

class Loop():
    def __init__(self, llm: LLMClient, 
                 tool_registry: ToolRegistry, 
                 ) -> None:
        self.client = llm