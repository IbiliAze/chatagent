"""Tests for CircuitBreaker's closed/open state transitions."""

import pytest

from app.error_handling.circuit_breaker import CircuitBreaker


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    """Build a CircuitBreaker with a low failure threshold for the tests."""
    circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=10)
    return circuit_breaker


class TestCircuitBreaker:
    """State transitions and pass-through behaviour of CircuitBreaker.call."""

    def test_runs_function_normally(self, circuit_breaker: CircuitBreaker) -> None:
        """A succeeding call passes through its result and leaves the breaker closed."""

        def evaluate():
            return 1 + 2

        response = circuit_breaker.call(evaluate)

        assert response is not None
        assert type(response) is int
        assert circuit_breaker.state == 'closed'
        assert circuit_breaker.failures == 0
        assert circuit_breaker.last_failure_time == 0

    def test_runs_function_fails(self, circuit_breaker: CircuitBreaker) -> None:
        """A single failure is recorded but does not open the breaker."""

        def evaluate():
            raise ValueError('boom')

        with pytest.raises(ValueError):
            circuit_breaker.call(evaluate)

        assert circuit_breaker.failures == 1
        assert circuit_breaker.state == 'closed'
        assert circuit_breaker.last_failure_time > 0

    def test_opens_after_threshold_failures(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Once failures reach the threshold, the breaker opens and rejects calls."""

        def evaluate():
            raise ValueError('boom')

        for _ in range(5):
            with pytest.raises(ValueError):
                circuit_breaker.call(evaluate)

        assert circuit_breaker.state == 'open'

        # further calls are rejected without invoking the function
        with pytest.raises(Exception, match='Circuit breaker is OPEN'):
            circuit_breaker.call(evaluate)
