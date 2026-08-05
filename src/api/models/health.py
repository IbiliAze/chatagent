"""Response model for the health check endpoint."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response"""

    status: Literal['healthy', 'degraded'] = 'healthy'
    environment: str
    version: str = '1.0.0'
    checks: dict[str, bool] = {}
