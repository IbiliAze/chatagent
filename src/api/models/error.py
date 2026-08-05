"""Standard error response model."""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: str
    detail: str | None = None
    request_id: str | None = None
