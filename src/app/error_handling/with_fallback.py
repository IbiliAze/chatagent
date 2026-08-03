from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

P = ParamSpec('P')
R = TypeVar('R')

FALLBACK_ERRORS = (
  APITimeoutError,
  APIConnectionError,
  RateLimitError,
  httpx.TimeoutException,
)


def with_fallback_llm(
  fallback: ChatOpenAI | ChatAnthropic,
) -> Callable[
  [Callable[P, Awaitable[R]]],
  Callable[P, Awaitable[R]],
]:
  def decorator(
    node: Callable[P, Awaitable[R]],
  ) -> Callable[P, Awaitable[R]]:
    @wraps(node)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
      try:
        return await node(*args, **kwargs)
      except FALLBACK_ERRORS:
        kwargs['llm'] = fallback
        return await node(*args, **kwargs)

    return wrapper

  return decorator
