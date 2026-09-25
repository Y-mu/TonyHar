import { describe, expect, it } from "vitest";

import { toThreadMessage, type UiMessage } from "./chat-messages";

describe("toThreadMessage", () => {
  it("does not attach assistant-only status to user messages", () => {
    const message: UiMessage = {
      id: "user-1",
      role: "user",
      content: "你好",
    };

    const converted = toThreadMessage(message);

    expect(converted).toEqual(message);
    expect(converted).not.toHaveProperty("status");
  });

  it("keeps status on assistant messages", () => {
    const converted = toThreadMessage({
      id: "assistant-1",
      role: "assistant",
      content: "你好",
      status: { type: "complete", reason: "stop" },
    });

    expect(converted.status).toEqual({ type: "complete", reason: "stop" });
  });
});
