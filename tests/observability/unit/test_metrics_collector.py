import pytest

from app.observability.metrics_collector import MetricsCollector


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    return MetricsCollector()


class TestMetricsCollector:
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

        assert summary.total_requests == 2
        assert summary.avg_latency_ms == 15
        assert summary.cache_hit_rate == 1
        assert summary.error_rate == 0.5
        assert summary.total_errors == 1
        assert summary.total_input_tokens == 150
        assert summary.total_output_tokens == 3000

    def test_empty_collector_returns_zeroed_summary(
        self, metrics_collector: MetricsCollector
    ) -> None:
        summary = metrics_collector.get_summary()

        assert summary.total_requests == 0
        assert summary.total_errors == 0
        assert summary.error_rate == 0
        assert summary.avg_latency_ms == 0
        assert summary.cache_hit_rate == 0
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0

    def test_error_rate_is_ratio_of_errors_to_requests(
        self, metrics_collector: MetricsCollector
    ) -> None:
        for error in (True, False, False, False):
            metrics_collector.record_request(
                latency_ms=1, input_tokens=1, output_tokens=1, error=error
            )

        assert metrics_collector.get_summary().error_rate == 0.25

    def test_cache_hit_rate_is_ratio_of_hits_to_total(
        self, metrics_collector: MetricsCollector
    ) -> None:
        for cache_hit in (True, False, False, False):
            metrics_collector.record_request(
                latency_ms=1, input_tokens=1, output_tokens=1, cache_hit=cache_hit
            )

        assert metrics_collector.get_summary().cache_hit_rate == 0.25

    def test_cache_hit_defaults_to_miss(
        self, metrics_collector: MetricsCollector
    ) -> None:
        metrics_collector.record_request(latency_ms=1, input_tokens=1, output_tokens=1)

        assert metrics_collector.get_summary().cache_hit_rate == 0

    def test_avg_latency_is_rounded_to_two_places(
        self, metrics_collector: MetricsCollector
    ) -> None:
        for latency_ms in (10, 20, 25):
            metrics_collector.record_request(
                latency_ms=latency_ms, input_tokens=1, output_tokens=1
            )

        assert metrics_collector.get_summary().avg_latency_ms == 18.33

    def test_failed_requests_still_count_tokens_and_latency(
        self, metrics_collector: MetricsCollector
    ) -> None:
        metrics_collector.record_request(
            latency_ms=40, input_tokens=100, output_tokens=200, error=True
        )

        summary = metrics_collector.get_summary()

        assert summary.total_requests == 1
        assert summary.total_errors == 1
        assert summary.avg_latency_ms == 40
        assert summary.total_input_tokens == 100
        assert summary.total_output_tokens == 200
