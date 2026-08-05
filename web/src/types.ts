export interface ChatRequest {
  message: string;
  thread_id: string;
}

export interface ChatResponse {
  response: string;
  thread_id: string;
  model_used: string;
  cached: boolean;
  processing_time_ms: number;
  timestamp: string;
}

export interface ApiErrorBody {
  detail: string;
}

export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: string;
  modelUsed?: string;
  cached?: boolean;
  processingTimeMs?: number;
  isError?: boolean;
}
