"""把纯意图分类结果映射为一种运行策略。"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from uuid import uuid4

from tonyhar.tooling import ToolRequest

from .user_intent_recognizer import IntentResult


class IntentRecognizer(Protocol):
    def classify(self, text: str) -> IntentResult:
        ...


class RunKind(str, Enum):
    DIRECT_TOOL = "direct_tool"
    RETRIEVAL_AGENT = "pre_tool_then_agent"
    AGENT = "agent"


@dataclass(frozen=True)
class RunCommand:
    """Dispatcher 的不可变输出，只描述本轮采用哪种运行策略。"""

    kind: RunKind
    route_name: str
    confidence: float | None = None
    tool_requests: tuple[ToolRequest, ...] = ()
    model_tool_names: tuple[str, ...] = ()


class RunDispatcher:
    """意图到运行策略的确定性映射，不执行 Tool 或模型。"""

    def __init__(self, recognizer: IntentRecognizer):
        self.recognizer = recognizer

    def dispatch(self, user_input: str) -> RunCommand:
        intent = self.recognizer.classify(user_input)

        if intent.name == "knowledge_list":
            return RunCommand(
                kind=RunKind.DIRECT_TOOL,
                route_name=intent.name,
                confidence=intent.score,
                tool_requests=(ToolRequest(
                    name="knowledge_list",
                    tool_call_id=f"routed_knowledge_list_{uuid4().hex}",
                ),),
            )

        if intent.name == "ingest":
            return RunCommand(
                kind=RunKind.DIRECT_TOOL,
                route_name=intent.name,
                confidence=intent.score,
                tool_requests=(ToolRequest(
                    name="file_ingestion",
                    arguments={"message": user_input},
                    tool_call_id=f"routed_file_ingestion_{uuid4().hex}",
                ),),
            )

        if intent.name == "knowledge_query":
            return RunCommand(
                kind=RunKind.RETRIEVAL_AGENT,
                route_name=intent.name,
                confidence=intent.score,
                tool_requests=(ToolRequest(
                    name="knowledge_search",
                    arguments={"query": user_input, "top_k": 5},
                    tool_call_id=f"routed_knowledge_search_{uuid4().hex}",
                ),),
            )

        return RunCommand(
            kind=RunKind.AGENT,
            route_name=intent.name,
            confidence=intent.score,
            model_tool_names=(
                "calculator",
                "knowledge_list",
                "knowledge_search",
            ),
        )
