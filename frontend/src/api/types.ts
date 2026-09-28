export type AgentEventType =
  | "run_started"
  | "intent_planned"
  | "model_started"
  | "text_delta"
  | "model_completed"
  | "tool_started"
  | "tool_completed"
  | "final_answer"
  | "run_failed";

export type AgentEvent = {
  type: AgentEventType;
  run_id: string;
  session_id: string;
  data: Record<string, unknown>;
};

export type SessionMessage = {
  role: "user" | "assistant";
  content: string;
};

export type SessionResponse = {
  session_id: string;
  messages: SessionMessage[];
};

export type CreateSessionResponse = {
  session_id: string;
};
