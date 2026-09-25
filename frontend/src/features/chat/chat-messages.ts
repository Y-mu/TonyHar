import type { ThreadMessageLike } from "@assistant-ui/react";

type UserUiMessage = {
  id: string;
  role: "user";
  content: string;
  status?: never;
};

type AssistantUiMessage = {
  id: string;
  role: "assistant";
  content: string;
  status?: ThreadMessageLike["status"];
};

export type UiMessage = UserUiMessage | AssistantUiMessage;

export function toThreadMessage(message: UiMessage): ThreadMessageLike {
  const common = {
    id: message.id,
    role: message.role,
    content: message.content,
  };
  return message.role === "assistant"
    ? { ...common, status: message.status }
    : common;
}
