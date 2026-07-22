from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from langsmith import traceable
from pydantic import BaseModel, Field

from api.main import app, limiter
from core.config.settings import get_settings


class ErrorResponse(BaseModel):
  """Standard error response"""

  error: str
  detail: str | None = None
  request_id: str | None = None


class ChatRequest(BaseModel):
  """Incoming chat request"""

  message: str = Field(
    ..., min_length=1, max_length=10_000, description="User's message to the agent"
  )

  thread_id: str = Field(default='default', description='Conversation thread ID')


class ChatResponse(BaseModel):
  """Chat response returned. to the client"""

  response: str
  thread_id: str
  model_used: str
  cached: bool = False
  processing_time_ms: float
  timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


settings = get_settings()


@app.post('/chat', response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
@traceable
async def chat(request: Request, body: ChatRequest):
  """Main chat endpoint"""

  with RequestTimer() as timer:
    security_notes = []

  return ChatResponse(cached)
