import os

import pytest
from cases import (
  SECURITY_CASES,
  SecurityRegressionCase,
)
from langchain_openai import ChatOpenAI

from app.security.security_guard import SecurityGuard

pytestmark = [
  pytest.mark.regression,
  pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='requires a real OPENAI_API_KEY'
  ),
]


@pytest.fixture
def security_guard() -> SecurityGuard:
  return SecurityGuard(ChatOpenAI(model='gpt-4o-mini', temperature=0))


class TestSecurityGuardRegression:
  """Locks in SecurityGuard's classification on a fixed set of known inputs.

  Rerun after changing the prompt or swapping models to catch silent drift.
  """

  @pytest.mark.parametrize('case', SECURITY_CASES)
  def test_known_input_still_classified_correctly(
    self, security_guard: SecurityGuard, case: SecurityRegressionCase
  ) -> None:
    result = security_guard.security_check(case['input'])

    assert result.safe is case['expect_safe']
