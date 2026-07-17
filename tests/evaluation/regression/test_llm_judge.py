import os

import pytest
from langchain_openai import ChatOpenAI

from common.evaluation.llm_judge import LLMJudge
from tests.evaluation.regression.cases import (
  JUDGE_CASES,
  JudgeRegressionCase,
)

pytestmark = [
  pytest.mark.regression,
  pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='requires a real OPENAI_API_KEY'
  ),
]


@pytest.fixture
def judge() -> LLMJudge:
  return LLMJudge(ChatOpenAI(model='gpt-4o-mini', temperature=0))


class TestLLMJudgeRegression:
  """Locks in LLMJudge's scoring behaviour on known good/bad responses."""

  @pytest.mark.parametrize('case', JUDGE_CASES)
  def test_known_response_still_scores_as_expected(
    self, judge: LLMJudge, case: JudgeRegressionCase
  ) -> None:
    result = judge.judge(case['question'], case['response'])

    if case['overall_at_least'] is not None:
      assert result.overall >= case['overall_at_least']
    if case['overall_at_most'] is not None:
      assert result.overall <= case['overall_at_most']
