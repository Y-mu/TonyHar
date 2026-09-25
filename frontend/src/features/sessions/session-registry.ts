const STORAGE_KEY = "tonyhar.sessions.v1";

export type SessionSummary = {
  id: string;
  title: string;
  updatedAt: string;
};

export function loadSessionRegistry(): SessionSummary[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is SessionSummary =>
        typeof item?.id === "string" &&
        typeof item?.title === "string" &&
        typeof item?.updatedAt === "string",
    );
  } catch {
    return [];
  }
}

export function saveSessionRegistry(sessions: SessionSummary[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

export function titleFromMessage(message: string): string {
  const normalized = message.trim().replace(/\s+/g, " ");
  return normalized.length > 24
    ? `${normalized.slice(0, 24)}…`
    : normalized || "新对话";
}
