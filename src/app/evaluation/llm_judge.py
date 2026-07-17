import json
from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable  # pyright: ignore[reportUnknownVariableType]


@dataclass(frozen=True)
class LLMJudgeResponse:
  correctness: int
  relevance: int
  clarity: int
  completeness: int
  overall: int
  error: Optional[str]


class LLMJudge:
  def __init__(self, llm: ChatOpenAI) -> None:
    self.llm = llm

  @traceable(name='judge_response')
  def judge(
    self, question: str, response: str, reference: Optional[str] = None
  ) -> LLMJudgeResponse:
    """Evaluate a response on multiple levels"""
    eval_prompt = ChatPromptTemplate.from_template("""
Evaluate this response on a scale of 1-10 for each criterion
                                                       
Question: {question}  
Response: {response}
{reference_section}       

Rate each criterion 1-10:
1. Correctness: Is the information accurate?
2. Relevance: Does it answer the question?
3. Clarity: Is it easy to understand?
4. Completeness: Does it fully address the question?   

Respond with ONLY this JSON object structure:
{{"correctness": X, "relevance": X, "clarity": X, "completeness": X, "overall": X }}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
""")

    reference_section = ''
    if reference:
      reference_section = f'Reference answer: {reference}'

    response_obj = self.llm.invoke(
      eval_prompt.format(
        question=question,
        response=response,
        reference_section=reference_section,
      )
    )

    if not isinstance(response_obj.content, str):
      return LLMJudgeResponse(
        error='Failed to parse LLM response',
        clarity=0,
        completeness=0,
        correctness=0,
        overall=0,
        relevance=0,
      )

    try:
      scores = json.loads(response_obj.content)
      return LLMJudgeResponse(
        error=None,
        clarity=scores['clarity'],
        completeness=scores['completeness'],
        correctness=scores['correctness'],
        overall=scores['overall'],
        relevance=scores['relevance'],
      )
    except json.JSONDecodeError:
      return LLMJudgeResponse(
        error='Failed to parse LLM response',
        clarity=0,
        completeness=0,
        correctness=0,
        overall=0,
        relevance=0,
      )
