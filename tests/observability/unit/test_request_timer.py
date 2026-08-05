"""Tests for RequestTimer's elapsed-time tracking."""

import time

import pytest

from app.observability.request_timer import RequestTimer


class TestRequestTimer:
    """RequestTimer's live vs. frozen elapsed_ms behaviour."""

    def test_measures_elapsed_time(self) -> None:
        """elapsed_ms reflects the wall-clock time spent inside the block."""
        with RequestTimer() as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms >= 10

    def test_elapsed_is_frozen_after_exit(self) -> None:
        """elapsed_ms stops advancing once the block has exited."""
        with RequestTimer() as timer:
            time.sleep(0.01)

        elapsed = timer.elapsed_ms
        time.sleep(0.01)

        assert timer.elapsed_ms == elapsed

    def test_elapsed_is_live_inside_block(self) -> None:
        """elapsed_ms keeps advancing while still inside the block."""
        with RequestTimer() as timer:
            first = timer.elapsed_ms
            time.sleep(0.01)
            second = timer.elapsed_ms

        assert second > first

    def test_records_elapsed_when_block_raises(self) -> None:
        """elapsed_ms is still recorded correctly when the block raises."""
        timer = RequestTimer()

        with pytest.raises(ValueError), timer:
            time.sleep(0.01)
            raise ValueError('boom')

        assert timer.elapsed_ms >= 10

    def test_elapsed_is_zero_before_start(self) -> None:
        """elapsed_ms is zero for a timer that was never started."""
        assert RequestTimer().elapsed_ms == 0.0
