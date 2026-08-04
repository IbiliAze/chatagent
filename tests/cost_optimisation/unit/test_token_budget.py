import pytest

from app.cost_optimisation.token_budget import TokenBudget


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
