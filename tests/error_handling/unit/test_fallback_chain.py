"""Tests for FallBackChain's model-to-model fallback behaviour."""

from unittest.mock import MagicMock

import pytest
from langchain.messages import AIMessage

from app.error_handling.fallback_chain import FallBackChain, InvokeResult


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set placeholder API keys so FallBackChain can construct its clients."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test')


@pytest.fixture
def fallback_chain() -> FallBackChain:
    """Build a FallBackChain instance for the tests."""
    return FallBackChain(temperature=1)


class TestFallBackChain:
    """FallBackChain.invoke's fallback-on-failure behaviour."""

    def test_fallback(self, fallback_chain: FallBackChain) -> None:
        """When the first model fails, invoke falls through to the next one."""
        failing = MagicMock()
        failing.invoke.side_effect = Exception('rate limited')

        succeeding = MagicMock()
        succeeding.invoke.return_value = AIMessage(content='ok')

        fallback_chain.models = [
            ('gpt-4o', failing),
            ('claude-sonnet', succeeding),
        ]

        result = fallback_chain.invoke('hello')

        assert result == InvokeResult(model_name='claude-sonnet', result='ok')
        failing.invoke.assert_called_once_with('hello')
        succeeding.invoke.assert_called_once_with('hello')
