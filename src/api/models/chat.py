from datetime import datetime, timezone

from pydantic import BaseModel, Field


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
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
