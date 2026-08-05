"""Tests for TokenBudget's estimation, budget checks, and usage tracking."""

import pytest

from app.cost_optimisation.token_budget import (
    CheckBudgetResponse,
    GetStatsResponse,
    TokenBudget,
)


@pytest.fixture
def token_budget() -> TokenBudget:
    """Build a TokenBudget with a small max_tokens_per_request for the tests."""
    return TokenBudget(max_tokens_per_request=100)


class TestEstimation:
    """estimate_tokens behaviour."""

    def test_returns_token_estimation(self, token_budget: TokenBudget) -> None:
        """estimate_tokens returns a positive integer for non-empty text."""
        estimate = token_budget.estimate_tokens('Hi how much will I cost?', 'gpt-4o')

        assert estimate is not None
        assert isinstance(estimate, int)
        assert estimate > 0


class TestBudget:
    """check_budget behaviour."""

    def test_outside_budget(self, token_budget: TokenBudget) -> None:
        """Text exceeding max_tokens_per_request is reported as over budget."""
        response = token_budget.check_budget('Hi how much will I cost?' * 100, 'gpt-4o')

        assert response is not None
        assert isinstance(response, CheckBudgetResponse)
        assert response.within_budget is False
        assert response.tokens > 0

    def test_within_budget(self, token_budget: TokenBudget) -> None:
        """Text within max_tokens_per_request is reported as within budget."""
        response = token_budget.check_budget('Hi how much will I cost?', 'gpt-4o')

        assert response is not None
        assert isinstance(response, CheckBudgetResponse)
        assert response.within_budget is True
        assert response.tokens > 0


class TestStats:
    """get_stats/record_usage behaviour."""

    def test_returns_token_estimation(self, token_budget: TokenBudget) -> None:
        """get_stats sums input, output, and request counts across record_usage calls."""
        token_budget.record_usage(input_tokens=50, output_tokens=500)
        token_budget.record_usage(input_tokens=25, output_tokens=100)

        stats = token_budget.get_stats()

        assert stats is not None
        assert isinstance(stats, GetStatsResponse)
        assert stats.total_input == 75
        assert stats.total_output == 600
        assert stats.total_requests == 2
