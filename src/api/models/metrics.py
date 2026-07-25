from pydantic import BaseModel


class MetricsResponse(BaseModel):
  """Metrics endpoint response"""

  total_requests: int
  total_errors: int
  error_state: str
  avg_latency_ms: float
  cache_hit_rate: int
  total_input_tokens: int
  total_output_tokens: int
