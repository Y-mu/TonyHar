"""Agent 领域事件到 SSE 传输事件的编码。"""

import json

from sse_starlette import ServerSentEvent

from tonyhar.agent.runnables import AgentEvent


def encode_agent_event(event: AgentEvent, sequence: int) -> ServerSentEvent:
    payload = {
        "type": event.type.value,
        "run_id": event.run_id,
        "session_id": event.session_id,
        "data": event.data,
    }
    return ServerSentEvent(
        id=f"{event.run_id}:{sequence}",
        event=event.type.value,
        data=json.dumps(payload, ensure_ascii=False, default=str),
    )


def encode_stream_error(
    *,
    run_id: str,
    session_id: str,
    sequence: int,
) -> ServerSentEvent:
    payload = {
        "type": "run_failed",
        "run_id": run_id,
        "session_id": session_id,
        "data": {
            "success": False,
            "stop_reason": "transport_error",
            "error_code": "web_stream_error",
            "error_message": "事件流异常中断",
        },
    }
    return ServerSentEvent(
        id=f"{run_id}:{sequence}",
        event="run_failed",
        data=json.dumps(payload, ensure_ascii=False),
    )
