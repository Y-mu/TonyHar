"""Agent 运行器抽象。"""

from abc import ABC, abstractmethod

from collections.abc import AsyncIterator

from .modes import AgentEvent, AgentResult, AgentRunContext


class AgentStateMachine(ABC):
    """Agent 状态机运行入口。"""

    @abstractmethod
    async def stream(
        self,
        context: AgentRunContext,
    ) -> AsyncIterator[AgentEvent]:
        raise NotImplementedError

    @abstractmethod
    async def invoke(
        self,
        context: AgentRunContext,
    ) -> AgentResult:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError
