from collections.abc import Callable
from time import time
from typing import Any, Literal, ParamSpec, TypeVar

type State = Literal['closed'] | Literal['open'] | Literal['half-open']

P = ParamSpec('P')
T = TypeVar('T')


class CircuitBreaker:
  """Circuit breaker for failing services"""

  state: State

  def __init__(
    self, failure_threshold: int = 5, recovery_timeout: float = 30.0
  ) -> None:
    self.failure_threshold = failure_threshold
    self.recovery_timeout = recovery_timeout
    self.failures = 0
    self.last_failure_time = 0
    self.state = 'closed'

  def call(self, func: Callable[P, T], *args: Any, **kwargs: Any):
    """Execute a function with circuit breaker protection"""

    # Check if circuit breaker should go from open to half-open
    if self.state == 'open':
      if time() - self.last_failure_time > self.recovery_timeout:
        self.state = 'half-open'
      else:
        raise Exception('Circuit breaker is OPEN')

    try:
      result = func(*args, **kwargs)

      if self.state == 'half-open':
        self.state = 'closed'
        self.failures = 0

      return result

    except Exception as e:
      self.failures += 1
      self.last_failure_time = time()

      if self.failures >= self.failure_threshold:
        self.state = 'open'

      raise e
