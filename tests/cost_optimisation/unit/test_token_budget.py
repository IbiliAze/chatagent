import pytest

from app.cost_optimisation.token_budget import (
  CheckBudgetResponse,
  GetStatsResponse,
  TokenBudget,
)


@pytest.fixture
def token_budget() -> TokenBudget:
  return TokenBudget(max_tokens_per_request=100)


class TestEstimation:
  def test_returns_token_estimation(self, token_budget: TokenBudget) -> None:
    estimate = token_budget.estimate_tokens('Hi how much will I cost?', 'gpt-4o')

    assert estimate is not None
    assert type(estimate) is int
    assert estimate > 0


class TestBudget:
  def test_outside_budget(self, token_budget: TokenBudget) -> None:
    response = token_budget.check_budget('Hi how much will I cost?' * 100, 'gpt-4o')

    assert response is not None
    assert isinstance(response, CheckBudgetResponse)
    assert response.within_budget is False
    assert response.tokens > 0

  def test_within_budget(self, token_budget: TokenBudget) -> None:
    response = token_budget.check_budget('Hi how much will I cost?', 'gpt-4o')

    assert response is not None
    assert isinstance(response, CheckBudgetResponse)
    assert response.within_budget is True
    assert response.tokens > 0


class TestStats:
  def test_returns_token_estimation(self, token_budget: TokenBudget) -> None:
    token_budget.record_usage(input_tokens=50, output_tokens=500)
    token_budget.record_usage(input_tokens=25, output_tokens=100)

    stats = token_budget.get_stats()

    assert stats is not None
    assert isinstance(stats, GetStatsResponse)
    assert stats.total_input == 75
    assert stats.total_output == 600
    assert stats.total_requests == 2
