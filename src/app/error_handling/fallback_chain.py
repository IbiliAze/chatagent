"""Fallback chain that tries multiple LLMs in order until one succeeds."""

from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langsmith import traceable  # pyright: ignore[reportUnknownVariableType]

type ModelName = str
type ModelType = list[tuple[ModelName, ChatOpenAI | ChatAnthropic]]


class AllModelsFailedError(Exception):
    """Raised when every model in the fallback chain fails to answer."""


@dataclass(frozen=True)
class InvokeResult:
    """The model that answered a query, and its response."""

    model_name: str
    result: str


class FallBackChain:
    """Try multiple models in order until one succeeds."""

    def __init__(self, temperature: float) -> None:
        self.models: ModelType = [
            ('gpt-4o', ChatOpenAI(model='gpt-4o', temperature=temperature, timeout=10)),
            (
                'claude-sonnet',
                ChatAnthropic(
                    model_name='claude-sonnet-4-5-20250929',
                    timeout=10,
                    stop=None,
                ),
            ),
            (
                'gpt-4o-mini',
                ChatOpenAI(model='gpt-4o-mini', temperature=temperature, timeout=10),
            ),
        ]

        self.cache: dict[str, str] = {}

    @traceable(name='fallback_invoke')
    def invoke(self, query: str, use_cache: bool = True) -> InvokeResult:
        """Answer the query, trying each model in order until one succeeds."""
        if use_cache and query in self.cache:
            return InvokeResult(model_name='cache', result=self.cache['query'])

        errors: list[str] = []
        for model_name, model in self.models:
            try:
                response = model.invoke(query)
                content = response.content
                result = content if isinstance(content, str) else str(content)
                self.cache[query] = result
                return InvokeResult(model_name=model_name, result=result)

            except Exception as e:  # pylint: disable=broad-exception-caught
                errors.append(f'{model_name}: {str(e)}')
                continue

        raise AllModelsFailedError(f'All models failed: {errors}')
