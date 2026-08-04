import pytest

from app.observability.metrics_collector import MetricsCollector, MetricsSummary


@pytest.fixture
def metrics_collector() -> MetricsCollector:
  return MetricsCollector()


class TestMetricsSummary:
  def test_records(self, metrics_collector: MetricsCollector) -> None:
    metrics_collector.record_request(
      latency_ms=10,
      input_tokens=100,
      output_tokens=1000,
      error=False,
      cache_hit=True,
    )
    metrics_collector.record_request(
      latency_ms=20,
      input_tokens=50,
      output_tokens=2000,
      error=True,
      cache_hit=True,
    )

    summary = metrics_collector.get_summary()

    assert summary is not None
    assert isinstance(summary, MetricsSummary)
    assert summary.total_requests == 2
    assert summary.avg_latency_ms == 15
    assert summary.cache_hit_rate == 1
    assert summary.error_rate == '0.5%'
    assert summary.total_errors == 1
    assert summary.total_input_tokens == 150
    assert summary.total_output_tokens == 3000
