"""Per-request token budget enforcement and usage tracking."""

from dataclasses import dataclass

import tiktoken

from core.config.settings import get_settings
from core.config.types import AvailableModels


@dataclass(frozen=True)
class CheckBudgetResponse:
    """Result of a budget check for a piece of text."""

    within_budget: bool
    tokens: int


@dataclass(frozen=True)
class GetStatsResponse:
    """Cumulative token usage across all recorded requests."""

    total_input: int
    total_output: int
    total_requests: int


settings = get_settings()


class TokenBudget:
    """Tracks token usage and checks text against the per-request budget."""

    def __init__(self, max_tokens_per_request: int = settings.token_budget) -> None:
        self.max_tokens_per_request = max_tokens_per_request
        self.usage = {
            'total_input': 0,
            'total_output': 0,
            'total_requests': 0,
        }

    def estimate_tokens(self, text: str, model: AvailableModels) -> int:
        """Token estimation with tiktoken"""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def check_budget(self, text: str, model: AvailableModels) -> CheckBudgetResponse:
        """Check if request is within budget"""
        tokens = self.estimate_tokens(text, model)
        return CheckBudgetResponse(
            within_budget=tokens <= self.max_tokens_per_request, tokens=tokens
        )

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage"""
        self.usage['total_input'] += input_tokens
        self.usage['total_output'] += output_tokens
        self.usage['total_requests'] += 1

    def get_stats(
        self,
    ) -> GetStatsResponse:
        """Get token usage stats"""
        return GetStatsResponse(
            total_input=self.usage['total_input'],
            total_output=self.usage['total_output'],
            total_requests=self.usage['total_requests'],
        )
