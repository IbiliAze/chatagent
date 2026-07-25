from fastapi import Request
from langsmith import traceable  # pyright: ignore[reportUnknownVariableType]

from api.main import app, limiter
from api.models.chat import ChatRequest, ChatResponse
from api.models.error import ErrorResponse
from core.config.settings import get_settings

settings = get_settings()


@app.post('/chat', response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
@traceable
async def chat(request: Request, body: ChatRequest):
  """Main chat endpoint"""

  with RequestTimer() as timer:
    security_notes = []

  return ChatResponse(cached)
