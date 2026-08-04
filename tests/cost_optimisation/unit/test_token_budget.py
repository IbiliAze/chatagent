import pytest

from app.cost_optimisation.token_budget import CheckBudgetResponse, TokenBudget


@pytest.fixture
def token_budget() -> TokenBudget:
  token_budget = TokenBudget(max_tokens_per_request=100)
  return token_budget


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
