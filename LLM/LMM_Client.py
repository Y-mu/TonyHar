from abc import ABC, abstractmethod
from typing import Sequence

from .LLMResponse import LLMResponse


class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict],
    ) -> LLMResponse:
        raise NotImplementedError