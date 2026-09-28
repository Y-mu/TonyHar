"""Agent 运行组件共享的事件构造函数。"""

from typing import Any

from .runnables import AgentEvent, AgentEventType, AgentRunContext


def build_event(
    context: AgentRunContext,
    event_type: AgentEventType,
    **data: Any,
) -> AgentEvent:
    payload = {"state": context.state.value}
    payload.update(data)
    return AgentEvent(
        type=event_type,
        run_id=context.run_id,
        session_id=context.session_id,
        data=payload,
    )
