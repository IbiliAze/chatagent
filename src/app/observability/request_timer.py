import time
from types import TracebackType
from typing import Self


class RequestTimer:
  """Measure the wall-clock duration of a block of work."""

  def __init__(self) -> None:
    self.start: float | None = None
    self.end: float | None = None

  def __enter__(self) -> Self:
    self.start = time.perf_counter()
    self.end = None
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
  ) -> None:
    self.end = time.perf_counter()

  @property
  def elapsed_ms(self) -> float:
    """Elapsed milliseconds; measured live while the timer is still running."""
    if self.start is None:
      return 0.0

    end = self.end if self.end is not None else time.perf_counter()
    return (end - self.start) * 1000
