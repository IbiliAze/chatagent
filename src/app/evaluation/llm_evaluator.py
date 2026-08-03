from typing import Literal, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable  # pyright: ignore[reportUnknownVariableType]
from langsmith.evaluation import (
  EvaluationResult,
  evaluate,  # pyright: ignore[reportUnknownVariableType]
)
from langsmith.schemas import Example, Run

type Input = dict[Literal['question'], str]
type Output = dict[Literal['answer'], str]


class LLMEvaluator:
  def __init__(self, llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_template(
      'Answer this question concisely: {question}'
    )
    self.chain = prompt | llm
    self.client = Client()

  def run_eval(self):
    evaluate(
      self._qa_target,
      data=self._create_eval_dataset(),
      evaluators=[self._correctness],
      experiment_prefix='qa-eval',
    )

  @traceable(name='qa_target')
  def _qa_target(self, input: Input) -> Output:
    response = self.chain.invoke({'question': input['question']})
    content = response.content
    return {'answer': content if isinstance(content, str) else str(content)}

  @staticmethod
  def _correctness(run: Run, example: Example | None) -> EvaluationResult:
    run_outputs = cast(
      Output,
      run.outputs or {},  # pyright: ignore[reportUnknownMemberType]
    )
    example_outputs = cast(
      Output,
      (example.outputs if example else None)  # pyright: ignore[reportUnknownMemberType]
      or {},
    )

    prediction = run_outputs.get('answer', '')
    reference = example_outputs.get('answer', '')

    score = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

    return EvaluationResult(key='correctness', score=score)

  def _create_eval_dataset(self) -> str:
    dataset_name = 'qa-eval-dataset'

    dataset = self.client.create_dataset(  # pyright: ignore[reportUnknownMemberType]
      dataset_name=dataset_name, description='Q&A evaluation dataset'
    )

    examples = [
      {
        'inputs': {'question': 'What is Python?'},
        'outputs': {'answer': 'Python is a high-level programming language'},
      },
      {
        'inputs': {'question': 'What is 6 * 8?'},
        'outputs': {'answer': '48'},
      },
    ]

    for ex in examples:
      self.client.create_example(
        inputs=ex['inputs'], outputs=ex['outputs'], dataset_id=dataset.id
      )

    print(f'Created a dataset with {len(examples)} datasets.')

    return dataset_name
