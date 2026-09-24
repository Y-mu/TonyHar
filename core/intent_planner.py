"""把语义路由结果转换为可执行但尚未执行的 Agent 计划。"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from uuid import uuid4

from tooling import ToolRequest

from .userIntentrecognizer import IntentResult


class IntentRecognizer(Protocol):
    def classify(self, text: str) -> IntentResult:
        ...


class ExecutionMode(str, Enum):
    DIRECT_TOOL = "direct_tool"
    PRE_TOOL_THEN_AGENT = "pre_tool_then_agent"
    AGENT = "agent"


@dataclass(frozen=True)
class IntentPlan:
    """纯路由计划，不读取或修改会话 Memory。"""

    mode: ExecutionMode
    user_message: str
    route_name: str
    confidence: float | None = None
    tool_calls: tuple[ToolRequest, ...] = ()


class IntentPlanner:
    def __init__(self, recognizer: IntentRecognizer):
        self.recognizer = recognizer

    def plan(self, user_input: str) -> IntentPlan:
        route = self.recognizer.classify(user_input)

        if route.name == "knowledge_list":
            return IntentPlan(
                mode=ExecutionMode.DIRECT_TOOL,
                user_message=user_input,
                route_name=route.name,
                confidence=route.score,
                tool_calls=(ToolRequest(
                    name="knowledge_list",
                    tool_call_id=f"routed_knowledge_list_{uuid4().hex}",
                ),),
            )

        if route.name == "ingest":
            return IntentPlan(
                mode=ExecutionMode.DIRECT_TOOL,
                user_message=user_input,
                route_name=route.name,
                confidence=route.score,
                tool_calls=(ToolRequest(
                    name="file_ingestion",
                    arguments={"message": user_input},
                    tool_call_id=f"routed_file_ingestion_{uuid4().hex}",
                ),),
            )

        confidence = (
            f", confidence={route.score:.3f}"
            if route.score is not None else ""
        )
        routed_message = (
            f"{user_input}\n\n[语义路由: {route.name}{confidence}]"
        )

        if route.name == "knowledge_query":
            routed_message += "\n请优先考虑从已入库的知识内容回答。"
            return IntentPlan(
                mode=ExecutionMode.PRE_TOOL_THEN_AGENT,
                user_message=routed_message,
                route_name=route.name,
                confidence=route.score,
                tool_calls=(ToolRequest(
                    name="knowledge_search",
                    arguments={"query": user_input, "top_k": 5},
                    tool_call_id=f"forced_knowledge_search_{uuid4().hex}",
                ),),
            )

        return IntentPlan(
            mode=ExecutionMode.AGENT,
            user_message=routed_message,
            route_name=route.name,
            confidence=route.score,
        )
