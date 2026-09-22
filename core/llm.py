from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
from abc import ABC, abstractmethod
import json
from openai import OpenAI


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class LLMResponse:
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: List[ToolCall] = field(default_factory=list)
        
    def get_tool_calls(self) -> List[ToolCall]:
        return self.tool_calls
    
    def get_tool_calls_as_dicts(self) -> List[dict]:
        return [call.to_openai() for call in self.tool_calls]
    
    def get_content(self) -> str:
        return self.content if self.content else ""


class BaseLLM(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict],
    ) -> LLMResponse:
        raise NotImplementedError
    
    def get_results(self):
        raise NotImplementedError




class DeepSeekLLM(BaseLLM):
    def __init__(self, 
                 api_key: str, 
                 base_url: str | None = "https://api.deepseek.com", 
                 model: str = "deepseek-flash", 
                 timeout: float = 60.0):
        self._model = model
        self.timeout  = timeout
        # Initialize other necessary attributes or configurations here
        
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url
            )
        
    async def chat(self, messages: Sequence[dict], tools: Sequence[dict]) -> LLMResponse:
        # Implement the chat functionality using the DeepSeek API
        response = self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            tools=list(tools) or None,
        )
        
        message = response.choices[0].message
        tool_calls = []
        
        for call in message.tool_calls or []:
            arguments = json.loads(call.function.arguments or "{}")

            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        usage = response.usage
        
        return LLMResponse(
                    content=message.content or "",
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    finish_reason=response.choices[0].finish_reason or "unknown",
                    tool_calls=tool_calls,
                )

    def get_results(self):
        # Implement a method to retrieve search results
        pass
    

    # Add more methods as needed for the DeepSeek client functionality


