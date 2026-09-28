import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Menu, MessageSquarePlus, RefreshCw, X } from "lucide-react";

import { createSession, getSession } from "./api/sessions";
import type { SessionMessage } from "./api/types";
import { ChatErrorBoundary } from "./features/chat/ChatErrorBoundary";
import { ChatRuntime } from "./features/chat/ChatRuntime";
import { initialRunViewState, type RunViewState } from "./features/chat/chat-events";
import {
  loadSessionRegistry,
  saveSessionRegistry,
  titleFromMessage,
  type SessionSummary,
} from "./features/sessions/session-registry";

type LoadState = "loading" | "ready" | "error";

export function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>(() => loadSessionRegistry());
  const initialSessionId = useRef(sessions[0]?.id ?? null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [initialMessages, setInitialMessages] = useState<SessionMessage[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [runView, setRunView] = useState<RunViewState>(initialRunViewState);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const rememberSession = useCallback((id: string, title = "新对话") => {
    setSessions((current) => {
      const existing = current.find((item) => item.id === id);
      const next = [
        { id, title: existing?.title ?? title, updatedAt: new Date().toISOString() },
        ...current.filter((item) => item.id !== id),
      ];
      saveSessionRegistry(next);
      return next;
    });
  }, []);

  const openSession = useCallback(async (id: string) => {
    setLoadState("loading");
    setError(null);
    try {
      const response = await getSession(id);
      setSessionId(response.session_id);
      setInitialMessages(response.messages);
      setRunView(initialRunViewState);
      rememberSession(response.session_id);
      setLoadState("ready");
      setSidebarOpen(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法加载会话");
      setLoadState("error");
    }
  }, [rememberSession]);

  const createNewSession = useCallback(async () => {
    setLoadState("loading");
    setError(null);
    try {
      const response = await createSession();
      setSessionId(response.session_id);
      setInitialMessages([]);
      setRunView(initialRunViewState);
      rememberSession(response.session_id);
      setLoadState("ready");
      setSidebarOpen(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法创建会话");
      setLoadState("error");
    }
  }, [rememberSession]);

  useEffect(() => {
    if (initialSessionId.current) void openSession(initialSessionId.current);
    else void createNewSession();
  }, [createNewSession, openSession]);

  const activeTitle = useMemo(
    () => sessions.find((session) => session.id === sessionId)?.title ?? "新对话",
    [sessionId, sessions],
  );

  const handleMessageSent = useCallback((message: string) => {
    if (!sessionId) return;
    setSessions((current) => {
      const currentItem = current.find((item) => item.id === sessionId);
      const next = current.map((item) =>
        item.id === sessionId
          ? {
              ...item,
              title: currentItem?.title === "新对话"
                ? titleFromMessage(message)
                : item.title,
              updatedAt: new Date().toISOString(),
            }
          : item,
      );
      saveSessionRegistry(next);
      return next;
    });
  }, [sessionId]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">跳转到对话内容</a>
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`} aria-label="会话导航">
        <div className="sidebar-header">
          <div className="brand-mark" aria-hidden="true">T</div>
          <div>
            <p className="eyebrow">TONYHAR</p>
            <h1>知识助手</h1>
          </div>
          <button className="icon-button sidebar-close" type="button" onClick={() => setSidebarOpen(false)} aria-label="关闭会话导航">
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <button className="new-session-button" type="button" onClick={() => void createNewSession()} disabled={loadState === "loading"}>
          <MessageSquarePlus size={18} aria-hidden="true" />
          新建对话
        </button>
        <div className="session-list-heading">
          <span>最近对话</span>
          <span className="session-count">{sessions.length}</span>
        </div>
        <nav className="session-list" aria-label="最近对话">
          {sessions.length === 0 ? (
            <p className="empty-sessions">还没有对话记录</p>
          ) : sessions.map((session) => (
            <button
              className={`session-item ${session.id === sessionId ? "session-item-active" : ""}`}
              key={session.id}
              type="button"
              onClick={() => void openSession(session.id)}
            >
              <MessageSquarePlus size={16} aria-hidden="true" />
              <span>{session.title}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">模块化单体 · 本地会话</div>
      </aside>
      {sidebarOpen && <button className="drawer-scrim" type="button" onClick={() => setSidebarOpen(false)} aria-label="关闭会话导航" />}

      <main className="main-panel" id="main-content">
        <header className="topbar">
          <button className="icon-button menu-button" type="button" onClick={() => setSidebarOpen(true)} aria-label="打开会话导航">
            <Menu size={20} aria-hidden="true" />
          </button>
          <div className="topbar-title">
            <span className="status-dot" aria-hidden="true" />
            <span>{activeTitle}</span>
          </div>
          <div className="topbar-actions">
            <span className={`run-label ${runView.terminal ? "run-label-terminal" : ""}`} role="status" aria-live="polite">{runView.label}</span>
          </div>
        </header>

        {runView.tools.length > 0 && (
          <div className="tool-strip" aria-live="polite" aria-label="工具执行状态">
            {runView.tools.map((tool) => <span className={`tool-chip tool-${tool.status}`} key={tool.id}>{tool.name} · {tool.status === "running" ? "运行中" : tool.status === "succeeded" ? "完成" : "失败"}</span>)}
          </div>
        )}
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button className="retry-button" type="button" onClick={() => sessionId ? void openSession(sessionId) : void createNewSession()}><RefreshCw size={15} aria-hidden="true" />重试</button>
          </div>
        )}
        {runView.error && !error && (
          <div className="error-banner" role="alert">
            <span>{runView.error}。你可以在输入框中再次发送。</span>
          </div>
        )}

        <section className="chat-stage" aria-label="聊天对话">
          {loadState === "loading" && <div className="loading-state" role="status"><span className="loading-spinner" aria-hidden="true" />正在准备会话…</div>}
          {loadState === "error" && !sessionId && <div className="empty-state"><p>暂时无法连接到 TonyHar。</p><button type="button" className="primary-button" onClick={() => void createNewSession()}>重新连接</button></div>}
          {loadState === "ready" && sessionId && (
            <ChatErrorBoundary
              resetKey={sessionId}
              onRetry={() => void openSession(sessionId)}
            >
              <ChatRuntime
                key={sessionId}
                sessionId={sessionId}
                initialMessages={initialMessages}
                onRunStateChange={setRunView}
                onMessageSent={handleMessageSent}
              />
            </ChatErrorBoundary>
          )}
        </section>
      </main>
    </div>
  );
}
