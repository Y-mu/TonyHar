import type { AgentEvent } from "../../api/types";

export type ToolActivity = {
  id: string;
  name: string;
  status: "running" | "succeeded" | "failed";
  error?: string;
};

export type RunViewState = {
  label: string;
  tools: ToolActivity[];
  error: string | null;
  answer: string | null;
  terminal: boolean;
};

export const initialRunViewState: RunViewState = {
  label: "准备就绪",
  tools: [],
  error: null,
  answer: null,
  terminal: false,
};

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

export function applyAgentEvent(
  state: RunViewState,
  event: AgentEvent,
): RunViewState {
  switch (event.type) {
    case "run_started":
      return {
        label: "正在开始本轮对话",
        tools: [],
        error: null,
        answer: null,
        terminal: false,
      };
    case "intent_planned":
      return { ...state, label: "已理解问题，正在规划回答" };
    case "model_started":
      return { ...state, label: "正在思考" };
    case "model_completed":
      return { ...state, label: "模型响应完成" };
    case "tool_started": {
      const id = text(event.data.tool_call_id, `${event.run_id}-tool`);
      const activity: ToolActivity = {
        id,
        name: text(event.data.name, "工具"),
        status: "running",
      };
      return {
        ...state,
        label: `正在使用 ${activity.name}`,
        tools: [...state.tools.filter((tool) => tool.id !== id), activity],
      };
    }
    case "tool_completed": {
      const result = (event.data.result ?? {}) as Record<string, unknown>;
      const id = text(result.tool_call_id, `${event.run_id}-tool`);
      const success = result.success === true;
      const existing = state.tools.find((tool) => tool.id === id);
      const activity: ToolActivity = {
        id,
        name: text(result.name, existing?.name ?? "工具"),
        status: success ? "succeeded" : "failed",
        error: success ? undefined : text(result.error_message, "工具执行失败"),
      };
      return {
        ...state,
        label: success ? `${activity.name} 已完成` : `${activity.name} 执行失败`,
        tools: [...state.tools.filter((tool) => tool.id !== id), activity],
      };
    }
    case "final_answer":
      return {
        ...state,
        label: "回答完成",
        answer: text(event.data.answer),
        terminal: true,
      };
    case "run_failed":
      return {
        ...state,
        label: "本轮运行失败",
        error: text(event.data.error_message, "暂时无法完成回答，请重试"),
        terminal: true,
      };
  }
}
