import type { UploadResponse, HealthResponse, SSEMessage, DocumentInfo } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = API_URL.replace(/^http/, "ws");

// ── Upload ──────────────────────────────────────────────────────────────────

export async function uploadFiles(
  files: File[]
): Promise<UploadResponse> {
  const formData = new FormData();

  if (files.length === 1) {
    formData.append("file", files[0]);
    const res = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Upload failed");
    }
    return res.json();
  }

  // batch
  for (const f of files) {
    formData.append("files", f);
  }
  const res = await fetch(`${API_URL}/upload/batch`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

// ── WebSocket for ingestion status ──────────────────────────────────────────

export function connectStatusWS(
  jobId: string,
  onMessage: (data: Record<string, unknown>) => void,
  onClose?: () => void
): WebSocket {
  const ws = new WebSocket(`${WS_URL}/ws/status/${jobId}`);
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      // ignore non-json
    }
  };
  ws.onclose = () => onClose?.();
  ws.onerror = () => ws.close();
  return ws;
}

// ── Streaming chat (SSE via fetch) ──────────────────────────────────────────

export async function* streamChat(
  query: string,
  k = 3,
  includeImages = true
): AsyncGenerator<SSEMessage> {
  const res = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k, include_images: includeImages }),
  });

  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.statusText}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      const payload = trimmed.slice(6);
      if (payload === "[DONE]") return;
      try {
        yield JSON.parse(payload) as SSEMessage;
      } catch {
        // skip malformed
      }
    }
  }
}

export async function clearChatHistory(): Promise<{ message: string; status: string; cleared_messages?: number }> {
  const res = await fetch(`${API_URL}/chat/clear`, {
    method: "POST",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to clear chat history");
  }

  return res.json();
}

// ── Image URL helper ────────────────────────────────────────────────────────

export function imageUrl(imageId: string): string {
  return `${API_URL}/images/${imageId}`;
}

// ── Health check ────────────────────────────────────────────────────────────

export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

// ── List uploaded documents ─────────────────────────────────────────────────

export async function fetchDocuments(): Promise<{ total: number; documents: DocumentInfo[] }> {
  const res = await fetch(`${API_URL}/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}
