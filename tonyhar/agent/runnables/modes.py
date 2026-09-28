"""Agent 单次运行使用的稳定协议模型。"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tonyhar.resilience import Deadline
from tonyhar.tooling import ToolRequest


class AgentState(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    CALLING_MODEL = "calling_model"
    EXECUTING_TOOLS = "executing_tools"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# 合法状态转换校验
_ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.CREATED: frozenset({
        AgentState.PLANNING,
        AgentState.FAILED,
        AgentState.CANCELLED,
    }),
    AgentState.PLANNING: frozenset({
        AgentState.CALLING_MODEL,
        AgentState.EXECUTING_TOOLS,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
    }),
    AgentState.CALLING_MODEL: frozenset({
        AgentState.EXECUTING_TOOLS,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
    }),
    AgentState.EXECUTING_TOOLS: frozenset({
        AgentState.CALLING_MODEL,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
    }),
    AgentState.COMPLETED: frozenset(),
    AgentState.FAILED: frozenset(),
    AgentState.CANCELLED: frozenset(),
}


@dataclass
class AgentRunContext:
    """一次 Agent 调用的可变状态，不跨会话共享。"""

    run_id: str
    session_id: str
    user_input: str
    messages: list[dict[str, Any]]
    deadline: Deadline
    state: AgentState = AgentState.CREATED
    step: int = 0
    pending_tools: list[ToolRequest] = field(default_factory=list)
    last_tool_results: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id 不能为空")
        if not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if not self.user_input.strip():
            raise ValueError("user_input 不能为空")
        if self.step < 0:
            raise ValueError("step 不能小于 0")
        if not isinstance(self.deadline, Deadline):
            raise TypeError("deadline 必须是 Deadline")
        self.messages = deepcopy(self.messages)
        self.pending_tools = list(self.pending_tools)
        self.last_tool_results = list(self.last_tool_results)

    def transition_to(self, next_state: AgentState) -> None:
        """校验并执行状态转换。"""
        if next_state is self.state:
            return
        allowed = _ALLOWED_TRANSITIONS[self.state]
        if next_state not in allowed:
            raise ValueError(
                f"非法 Agent 状态转换: {self.state.value} -> "
                f"{next_state.value}"
            )
        self.state = next_state

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            **metadata,
        })


class AgentEventType(str, Enum):
    RUN_STARTED = "run_started"
    INTENT_PLANNED = "intent_planned"
    MODEL_STARTED = "model_started"
    TEXT_DELTA = "text_delta"
    MODEL_COMPLETED = "model_completed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    FINAL_ANSWER = "final_answer"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True)
class AgentEvent:
    """第三阶段事件流将使用的供应商无关事件。"""

    type: AgentEventType
    run_id: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    """一次 Agent 运行的结构化终态。"""

    run_id: str
    session_id: str
    success: bool
    answer: str = ""
    stop_reason: str = "completed"
    steps: int = 0
    error_code: str | None = None
    error_message: str | None = None
