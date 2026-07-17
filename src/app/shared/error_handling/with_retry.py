from collections.abc import Callable
from functools import wraps
from time import sleep
from typing import ParamSpec, TypeVar

P = ParamSpec('P')
T = TypeVar('T')


def with_retry(
  max_attempts: int = 5, initial_backoff: float = 1.0, max_backoff: float = 30.0
) -> Callable[[Callable[P, T]], Callable[P, T]]:
  """Retry a function with exponential backoff on exception.

  Example:
    @with_retry(max_attempts=3, initial_backoff=1.0, max_backoff=30.0)
    def call_api(x: int, y: int) -> dict:
      return requests.get(f'https://api.example.com/{x}/{y}').json()

    call_api(1, 2)  # retried transparently on exception, same call signature
  """

  def decorator(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
      backoff = initial_backoff
      last_error: Exception | None = None

      for attempt in range(1, max_attempts + 1):
        try:
          return func(*args, **kwargs)
        except Exception as e:
          last_error = e
          if attempt == max_attempts:
            break
          sleep(backoff)
          backoff = min(backoff * 2, max_backoff)

      assert last_error is not None
      raise last_error

    return wrapper

  return decorator
