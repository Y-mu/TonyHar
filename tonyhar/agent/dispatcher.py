"""把纯意图分类结果映射为一种运行策略。"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from tonyhar.tooling import ToolRequest

from .user_intent_recognizer import IntentResult


_URL_PATTERN = re.compile(r"https?://[^\s<>\"'）】》]+", re.IGNORECASE)
_URL_TRAILING_PUNCTUATION = "，。！？、；：,.!?;:)]}>'\""


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

        if intent.name == "spider_url":
            url = self._extract_url(user_input)
            return RunCommand(
                kind=RunKind.DIRECT_TOOL,
                route_name=intent.name,
                confidence=intent.score,
                tool_requests=(ToolRequest(
                    name="spider_url",
                    arguments={"url": url},
                    tool_call_id=f"routed_spider_url_{uuid4().hex}",
                ),),
            )

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

    @staticmethod
    def _extract_url(user_input: str) -> str:
        """从用户指令中提取并校验一个 HTTP(S) URL。"""
        match = _URL_PATTERN.search(user_input)
        if match is None:
            raise ValueError("抓取网页需要提供 http 或 https URL")

        url = match.group(0).rstrip(_URL_TRAILING_PUNCTUATION)
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("网页地址必须是有效的 http 或 https URL")
        return url
