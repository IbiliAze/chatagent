import time

import pytest

from app.observability.request_timer import RequestTimer


class TestRequestTimer:
  def test_measures_elapsed_time(self) -> None:
    with RequestTimer() as timer:
      time.sleep(0.01)

    assert timer.elapsed_ms >= 10

  def test_elapsed_is_frozen_after_exit(self) -> None:
    with RequestTimer() as timer:
      time.sleep(0.01)

    elapsed = timer.elapsed_ms
    time.sleep(0.01)

    assert timer.elapsed_ms == elapsed

  def test_elapsed_is_live_inside_block(self) -> None:
    with RequestTimer() as timer:
      first = timer.elapsed_ms
      time.sleep(0.01)
      second = timer.elapsed_ms

    assert second > first

  def test_records_elapsed_when_block_raises(self) -> None:
    timer = RequestTimer()

    with pytest.raises(ValueError), timer:
      time.sleep(0.01)
      raise ValueError('boom')

    assert timer.elapsed_ms >= 10

  def test_elapsed_is_zero_before_start(self) -> None:
    assert RequestTimer().elapsed_ms == 0.0
