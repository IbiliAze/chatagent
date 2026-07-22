from dataclasses import asdict

from pydantic import BaseModel

from api.main import app, metrics


class MetricsResponse(BaseModel):
  """Metrics endpoint response"""

  total_requests: int
  total_errors: int
  error_state: str
  avg_latency_ms: float
  cache_hit_rate: int
  total_input_tokens: int
  total_output_tokens: int


@app.post('/metrics', response_model=MetricsResponse)
async def metrics_summary():
  """Get metrics summary"""

  summary = metrics.get_summary()
  return MetricsResponse(**asdict(summary))
