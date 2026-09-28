import { describe, expect, it } from "vitest";

import type { AgentEvent } from "../../api/types";
import { applyAgentEvent, initialRunViewState } from "./chat-events";

function event(type: AgentEvent["type"], data: Record<string, unknown>) {
  return {
    type,
    run_id: "run-1",
    session_id: "session-1",
    data,
  } satisfies AgentEvent;
}

describe("applyAgentEvent", () => {
  it("tracks tool lifecycle", () => {
    const running = applyAgentEvent(
      initialRunViewState,
      event("tool_started", {
        tool_call_id: "tool-1",
        name: "knowledge_search",
      }),
    );
    expect(running.tools[0]).toMatchObject({
      id: "tool-1",
      name: "knowledge_search",
      status: "running",
    });

    const completed = applyAgentEvent(
      running,
      event("tool_completed", {
        result: {
          tool_call_id: "tool-1",
          name: "knowledge_search",
          success: true,
        },
      }),
    );
    expect(completed.tools[0].status).toBe("succeeded");
  });

  it("captures terminal answer and failure", () => {
    const completed = applyAgentEvent(
      initialRunViewState,
      event("final_answer", { answer: "完成" }),
    );
    expect(completed).toMatchObject({ answer: "完成", terminal: true });

    const failed = applyAgentEvent(
      initialRunViewState,
      event("run_failed", { error_message: "模型不可用" }),
    );
    expect(failed).toMatchObject({ error: "模型不可用", terminal: true });
  });

  it("shows text generation while deltas arrive", () => {
    const streaming = applyAgentEvent(
      initialRunViewState,
      event("text_delta", { delta: "你" }),
    );

    expect(streaming).toMatchObject({
      label: "正在生成回答",
      terminal: false,
    });
  });
});
