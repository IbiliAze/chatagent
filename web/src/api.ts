import type { ApiErrorBody, ChatRequest, ChatResponse } from './types';

export class ChatApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ChatApiError';
    this.status = status;
  }
}

export async function sendChatMessage(
  request: ChatRequest,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  let res: Response;
  try {
    res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: signal ?? null,
    });
  } catch {
    throw new ChatApiError(0, 'Could not reach the API. Is the backend running on :8000?');
  }

  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body = (await res.json()) as ApiErrorBody;
      if (body.detail) detail = body.detail;
    } catch {
      // response wasn't JSON - fall back to the generic message above
    }
    throw new ChatApiError(res.status, detail);
  }

  return (await res.json()) as ChatResponse;
}
