from typing import Sequence
import os
import json

from .LMM_Client import LLMClient
from .LLMResponse import LLMResponse,ToolCall
from openai import OpenAI




class DeepSeek_Client(LLMClient):
    def __init__(self, 
                 api_key: str, 
                 base_url: str, 
                 model: str = "deepseek-flash", 
                 timeout: float = 60.0):
        self.model = model
        self.timeout  = timeout
        # Initialize other necessary attributes or configurations here
        
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url)
        

    async def chat(self, messages: Sequence[dict], tools: Sequence[dict]) -> LLMResponse:
        # Implement the chat functionality using the DeepSeek API
        response = await self._client.chat.completions.create(
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

