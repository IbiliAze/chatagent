"""Regression tests locking in SecurityGuard's classification on known inputs."""

import os

import pytest
from langchain_openai import ChatOpenAI

from app.security.security_guard import SecurityGuard
from tests.security.regression.cases import (
    SECURITY_CASES,
    SecurityRegressionCase,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(
        not os.environ.get('OPENAI_API_KEY'), reason='requires a real OPENAI_API_KEY'
    ),
]


@pytest.fixture
def security_guard() -> SecurityGuard:
    """Build a SecurityGuard backed by a real gpt-4o-mini call."""
    return SecurityGuard(ChatOpenAI(model='gpt-4o-mini', temperature=0))


class TestSecurityGuardRegression:
    """Locks in SecurityGuard's classification on a fixed set of known inputs.

    Rerun after changing the prompt or swapping models to catch silent drift.
    """

    @pytest.mark.parametrize('case', SECURITY_CASES)
    def test_known_input_still_classified_correctly(
        self, security_guard: SecurityGuard, case: SecurityRegressionCase
    ) -> None:
        """A known input still classifies with its expected safe/unsafe verdict."""
        result = security_guard.security_check(case['input'])

        assert result.safe is case['expect_safe']
