import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useExternalStoreRuntime,
  type AppendMessage,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import remarkGfm from "remark-gfm";
import { Send, Square } from "lucide-react";

import { streamChat } from "../../api/chat-stream";
import type { SessionMessage } from "../../api/types";
import { applyAgentEvent, initialRunViewState, type RunViewState } from "./chat-events";
import { toThreadMessage, type UiMessage } from "./chat-messages";

export type { UiMessage } from "./chat-messages";

type ChatRuntimeProps = {
  sessionId: string;
  initialMessages: SessionMessage[];
  onRunStateChange: (state: RunViewState) => void;
  onMessageSent?: (message: string) => void;
};

function messageId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

function textFromAppendMessage(message: AppendMessage): string {
  const content = message.content as unknown;
  if (typeof content === "string") return content.trim();
  if (!Array.isArray(content)) return "";
  return content
    .filter((part): part is { type: "text"; text: string } =>
      typeof part === "object" && part !== null && (part as { type?: unknown }).type === "text" && typeof (part as { text?: unknown }).text === "string",
    )
    .map((part) => part.text)
    .join("")
    .trim();
}

function initialUiMessages(messages: SessionMessage[]): UiMessage[] {
  return messages.map((message, index) =>
    message.role === "assistant"
      ? {
          id: `history-${index}`,
          role: "assistant",
          content: message.content,
          status: { type: "complete", reason: "stop" },
        }
      : {
          id: `history-${index}`,
          role: "user",
          content: message.content,
        },
  );
}

export function ChatRuntime({
  sessionId,
  initialMessages,
  onRunStateChange,
  onMessageSent,
}: ChatRuntimeProps) {
  const [messages, setMessages] = useState<UiMessage[]>(() => initialUiMessages(initialMessages));
  const [isRunning, setIsRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const assistantIdRef = useRef<string | null>(null);
  const updateMessages: Dispatch<SetStateAction<UiMessage[]>> = useCallback(
    (update) => setMessages(update),
    [],
  );

  const onNew = useCallback(
    async (append: AppendMessage) => {
      const content = textFromAppendMessage(append);
      if (!content || isRunning) return;

      const userId = append.sourceId ?? messageId("user");
      const assistantId = messageId("assistant");
      assistantIdRef.current = assistantId;
      abortRef.current = new AbortController();
      setIsRunning(true);
      onRunStateChange(initialRunViewState);
      onMessageSent?.(content);

      updateMessages((current) => [
        ...current,
        {
          id: userId,
          role: "user",
          content,
        },
        {
          id: assistantId,
          role: "assistant",
          content: "",
          status: { type: "running" },
        },
      ]);

      let runState = initialRunViewState;
      try {
        await streamChat({
          sessionId,
          message: content,
          signal: abortRef.current.signal,
          onEvent: (event) => {
            runState = applyAgentEvent(runState, event);
            onRunStateChange(runState);
            if (event.type === "text_delta") {
              const delta = typeof event.data.delta === "string" ? event.data.delta : "";
              if (delta) {
                updateMessages((current) =>
                  current.map((item) =>
                    item.role === "assistant" && item.id === assistantId
                      ? { ...item, content: item.content + delta, status: { type: "running" } }
                      : item,
                  ),
                );
              }
            }
            if (event.type === "final_answer") {
              const answer = runState.answer ?? "";
              updateMessages((current) =>
                current.map((item) =>
                  item.role === "assistant" && item.id === assistantId
                    ? { ...item, content: answer, status: { type: "complete", reason: "stop" } }
                    : item,
                ),
              );
            }
            if (event.type === "run_failed") {
              const error = runState.error ?? "暂时无法完成回答，请重试";
              updateMessages((current) =>
                current.map((item) =>
                  item.role === "assistant" && item.id === assistantId
                    ? { ...item, content: error, status: { type: "incomplete", reason: "error", error } }
                    : item,
                ),
              );
            }
          },
        });
        if (!runState.terminal) {
          const message = "连接已结束，但服务端没有返回最终结果";
          const failedState = { ...runState, label: "本轮运行失败", error: message, terminal: true };
          onRunStateChange(failedState);
          updateMessages((current) =>
            current.map((item) =>
              item.role === "assistant" && item.id === assistantId
                ? { ...item, content: message, status: { type: "incomplete", reason: "error", error: message } }
                : item,
            ),
          );
        }
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          const message = error instanceof Error ? error.message : "请求失败，请重试";
          const failedState = { ...runState, label: "本轮运行失败", error: message, terminal: true };
          onRunStateChange(failedState);
          updateMessages((current) =>
            current.map((item) =>
              item.role === "assistant" && item.id === assistantId
                ? { ...item, content: message, status: { type: "incomplete", reason: "error", error: message } }
                : item,
            ),
          );
        }
      } finally {
        setIsRunning(false);
        abortRef.current = null;
        assistantIdRef.current = null;
      }
    },
    [isRunning, onMessageSent, onRunStateChange, sessionId, updateMessages],
  );

  const onCancel = useCallback(async () => {
    abortRef.current?.abort();
    const assistantId = assistantIdRef.current;
    if (assistantId) {
      updateMessages((current) =>
        current.map((item) =>
          item.role === "assistant" && item.id === assistantId
            ? { ...item, content: item.content || "已停止生成。", status: { type: "incomplete", reason: "cancelled", error: "已停止生成" } }
            : item,
        ),
      );
    }
    onRunStateChange({ ...initialRunViewState, label: "已停止", terminal: true });
  }, [onRunStateChange, updateMessages]);

  const runtime = useExternalStoreRuntime<UiMessage>({
    messages,
    convertMessage: toThreadMessage,
    setMessages: (next) => updateMessages(next as UiMessage[]),
    isRunning,
    isSendDisabled: isRunning,
    onNew,
    onCancel,
  });

  useEffect(() => () => abortRef.current?.abort(), []);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread-root">
        <ThreadPrimitive.Viewport className="thread-viewport">
          <ThreadPrimitive.Empty>
            <div className="welcome-state">
              <div className="welcome-mark" aria-hidden="true">T</div>
              <p className="eyebrow">TONYHAR AGENT</p>
              <h2>今天想了解什么？</h2>
              <p>我可以基于本地知识库回答问题，并在需要时调用工具完成任务。</p>
            </div>
          </ThreadPrimitive.Empty>
          <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
          <ThreadPrimitive.ViewportFooter className="composer-footer">
            <Composer />
          </ThreadPrimitive.ViewportFooter>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="message-row message-row-user">
      <div className="message-bubble message-bubble-user">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="message-row message-row-assistant">
      <div className="message-avatar" aria-hidden="true">T</div>
      <div className="message-content">
        <MessagePrimitive.If hasContent={false}>
          <div className="thinking-dots" role="status" aria-label="正在生成回答"><span /><span /><span /></div>
        </MessagePrimitive.If>
        <MessagePrimitive.Parts
          components={{
            Text: () => <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} smooth={false} />,
          }}
        />
        <MessagePrimitive.Error>
          <span className="message-inline-error">生成未完成，请重试。</span>
        </MessagePrimitive.Error>
      </div>
    </MessagePrimitive.Root>
  );
}

function Composer() {
  return (
    <ComposerPrimitive.Root className="composer">
      <label className="sr-only" htmlFor="chat-composer">发送消息</label>
      <ComposerPrimitive.Input
        id="chat-composer"
        className="composer-input"
        placeholder="输入你的问题…"
        name="message"
        submitMode="enter"
        rows={1}
      />
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel className="icon-button composer-action" aria-label="停止生成" title="停止生成">
          <Square size={16} fill="currentColor" aria-hidden="true" />
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send className="icon-button composer-action" aria-label="发送消息" title="发送消息">
          <Send size={18} aria-hidden="true" />
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
    </ComposerPrimitive.Root>
  );
}
