"""Response model for the metrics endpoint."""

from pydantic import BaseModel


class MetricsResponse(BaseModel):
    """Metrics endpoint response"""

    total_requests: int
    total_errors: int
    error_rate: float
    avg_latency_ms: float
    cache_hit_rate: float
    total_input_tokens: int
    total_output_tokens: int
