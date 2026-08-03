from collections.abc import Callable
from typing import cast

import httpx
import openai
from anthropic import (
  APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
  APITimeoutError as AnthropicAPITimeoutError,
)
from anthropic import (
  InternalServerError as AnthropicInternalServerError,
)
from anthropic import (
  RateLimitError as AnthropicRateLimitError,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableWithFallbacks
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import BaseModel

from core.config.settings import get_settings

FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
  openai.APITimeoutError,
  openai.APIConnectionError,
  openai.RateLimitError,
  openai.InternalServerError,
  AnthropicAPITimeoutError,
  AnthropicAPIConnectionError,
  AnthropicRateLimitError,
  AnthropicInternalServerError,
  httpx.TimeoutException,
)


class Models:
  def __init__(self) -> None:
    settings = get_settings()

    self.primary_llm = ChatOpenAI(
      model=settings.primary_model,
      api_key=settings.openai_api_key,  # pyright: ignore[reportArgumentType]
      temperature=0,
      max_retries=0,
      timeout=30,
    )

    fallback_llm = ChatAnthropic(
      model_name=settings.fallback_model,
      api_key=settings.anthropic_api_key,  # pyright: ignore[reportArgumentType]
      temperature=0,
      stop=None,
      max_retries=0,
      timeout=30,
    )

    fallback_llm_2 = ChatOpenAI(
      model=settings.fallback_model_2,
      api_key=settings.openai_api_key,  # pyright: ignore[reportArgumentType]
      temperature=0,
      max_retries=0,
      timeout=30,
    )

    self._chain: list[BaseChatModel] = [self.primary_llm, fallback_llm, fallback_llm_2]

    self.llm = self._compose(
      lambda model: cast(Runnable[LanguageModelInput, AIMessage], model)
    )

    self.embedding_llm = OpenAIEmbeddings(
      model=settings.embedding_model,
      max_retries=0,
      timeout=30,
    )

  def _compose[OutputT](
    self,
    configure: Callable[[BaseChatModel], Runnable[LanguageModelInput, OutputT]],
  ) -> RunnableWithFallbacks[LanguageModelInput, OutputT]:
    """Apply `configure` to every model in the chain, then wire up the fallbacks.

    Binding has to happen underneath the fallback wrapper: bind_tools and
    with_structured_output live on BaseChatModel, and RunnableWithFallbacks does
    not re-expose them.
    """
    primary, *rest = self._chain
    return configure(primary).with_fallbacks(
      [configure(model) for model in rest],
      exceptions_to_handle=FALLBACK_ERRORS,
    )

  def with_tools(
    self, tools: list[BaseTool]
  ) -> RunnableWithFallbacks[LanguageModelInput, AIMessage]:
    return self._compose(lambda model: model.bind_tools(tools))

  def with_schema[SchemaT: BaseModel](
    self, schema: type[SchemaT]
  ) -> RunnableWithFallbacks[LanguageModelInput, SchemaT]:
    return self._compose(
      lambda model: cast(
        Runnable[LanguageModelInput, SchemaT],
        model.with_structured_output(schema),
      )
    )
