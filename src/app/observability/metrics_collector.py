from dataclasses import dataclass


@dataclass(frozen=True)
class MetricsSummary:
  total_requests: int
  total_errors: int
  error_rate: str
  avg_latency_ms: float
  total_input_tokens: float
  total_output_tokens: float
  cache_hit_rate: float


class MetricsCollector:
  """Collect and aggregate metrics."""

  def __init__(self) -> None:
    self.metrics: dict[str, float] = {
      'requests_total': 0,
      'errors_total': 0,
      'latecy_sum': 0,
      'latency_count': 0,
      'tokens_input': 0,
      'tokens_output': 0,
      'cache_hits': 0,
      'cache_misses': 0,
    }

  def record_request(
    self,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    error: bool = False,
    cache_hit: bool = False,
  ):
    self.metrics['requests_total'] += 1
    self.metrics['latecy_sum'] += latency_ms
    self.metrics['latency_count'] += 1
    self.metrics['tokens_input'] += input_tokens
    self.metrics['tokens_output'] += output_tokens

    if error:
      self.metrics['errors_total'] += 1

    if cache_hit:
      self.metrics['cache_hits'] += 1
    else:
      self.metrics['cache_misses'] += 1

  def get_summary(self) -> MetricsSummary:
    """Get metrics summary."""
    avg_latency = (
      self.metrics['latecy_sum'] / self.metrics['latency_count']
      if self.metrics['latency_count'] > 0
      else 0
    )

    error_rate = (
      self.metrics['errors_total'] / self.metrics['requests_total']
      if self.metrics['requests_total'] > 0
      else 0
    )

    cache_hit_rate = (
      self.metrics['cache_hits']
      / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
      if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0
      else 0
    )

    return MetricsSummary(
      total_requests=int(self.metrics['requests_total']),
      total_errors=int(self.metrics['errors_total']),
      avg_latency_ms=round(avg_latency, 2),
      error_rate=f'{error_rate}:.2%',
      cache_hit_rate=cache_hit_rate,
      total_input_tokens=self.metrics['tokens_input'],
      total_output_tokens=self.metrics['tokens_output'],
    )
