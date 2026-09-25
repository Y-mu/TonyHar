import { requestJson } from "./http";
import type { CreateSessionResponse, SessionResponse } from "./types";

export function createSession(): Promise<CreateSessionResponse> {
  return requestJson<CreateSessionResponse>("/api/v1/sessions", {
    method: "POST",
  });
}

export function getSession(sessionId: string): Promise<SessionResponse> {
  return requestJson<SessionResponse>(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}`,
  );
}

export function getReadiness(): Promise<{ status: string }> {
  return requestJson<{ status: string }>("/health/ready");
}
