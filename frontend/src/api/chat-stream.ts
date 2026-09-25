import { EventSourceParserStream } from "eventsource-parser/stream";

import { API_BASE_URL, ApiError } from "./http";
import type { AgentEvent, AgentEventType } from "./types";

const EVENT_TYPES = new Set<AgentEventType>([
  "run_started",
  "intent_planned",
  "model_started",
  "model_completed",
  "tool_started",
  "tool_completed",
  "final_answer",
  "run_failed",
]);

function parseAgentEvent(data: string): AgentEvent {
  const value = JSON.parse(data) as Partial<AgentEvent>;
  if (
    typeof value.type !== "string" ||
    !EVENT_TYPES.has(value.type as AgentEventType) ||
    typeof value.run_id !== "string" ||
    typeof value.session_id !== "string" ||
    value.data === null ||
    typeof value.data !== "object"
  ) {
    throw new Error("服务端返回了无效的 AgentEvent");
  }
  return value as AgentEvent;
}

export async function streamChat(options: {
  sessionId: string;
  message: string;
  signal: AbortSignal;
  onEvent: (event: AgentEvent) => void;
}): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/sessions/${encodeURIComponent(options.sessionId)}/messages/stream`,
    {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: options.message }),
      signal: options.signal,
    },
  );

  if (!response.ok) {
    let message = `发送失败（${response.status}）`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // 保留稳定的 HTTP 错误信息。
    }
    throw new ApiError(message, response.status);
  }

  if (!response.body) {
    throw new Error("浏览器没有提供响应流");
  }

  const events = response.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(
      new EventSourceParserStream({
        maxBufferSize: 1_048_576,
        onError: "terminate",
      }),
    );

  const reader = events.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      options.onEvent(parseAgentEvent(value.data));
    }
  } finally {
    reader.releaseLock();
  }
}
