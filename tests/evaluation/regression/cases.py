from typing import Optional, TypedDict


class JudgeRegressionCase(TypedDict):
  question: str
  response: str
  overall_at_least: Optional[int]
  overall_at_most: Optional[int]


JUDGE_CASES: list[JudgeRegressionCase] = [
  {
    'question': 'What is the capital of France?',
    'response': 'The capital of France is Paris.',
    'overall_at_least': 8,
    'overall_at_most': None,
  },
  {
    'question': 'What is the capital of France?',
    'response': 'I like turtles.',
    'overall_at_least': None,
    'overall_at_most': 3,
  },
]
