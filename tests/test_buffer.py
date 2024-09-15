"""Tests for ring buffer."""
from dravex.core.buffer import RingBuffer
import pytest

class TestRingBuffer:
    def test_push_pop(self):
        b = RingBuffer(5)
        b.push(1.0)
        b.push(2.0)
        assert b.pop() == 1.0

    def test_capacity(self):
        b = RingBuffer(3)
        for i in range(5):
            b.push(float(i))
        assert b.count == 3

    def test_read_all(self):
        b = RingBuffer(4)
        for i in range(4):
            b.push(float(i))
        assert b.read_all() == [0.0, 1.0, 2.0, 3.0]

    def test_overflow_wraps(self):
        b = RingBuffer(3)
        b.push_many([1, 2, 3, 4, 5])
        assert b.read_all() == [3.0, 4.0, 5.0]

    def test_is_empty(self):
        b = RingBuffer(5)
        assert b.is_empty

    def test_is_full(self):
        b = RingBuffer(2)
        b.push(1)
        b.push(2)
        assert b.is_full

    def test_running_mean(self):
        b = RingBuffer(4)
        b.push_many([2, 4, 6, 8])
        assert abs(b.running_mean() - 5.0) < 1e-10

    def test_latest(self):
        b = RingBuffer(10)
        b.push_many(range(10))
        assert b.latest(3) == [7.0, 8.0, 9.0]

    def test_clear(self):
        b = RingBuffer(5)
        b.push_many([1, 2, 3])
        b.clear()
        assert b.is_empty


class TestSlidingWindow:
    def test_push_mean(self):
        from dravex.core.buffer import SlidingWindow
        sw = SlidingWindow(5)
        for i in range(5):
            sw.push(float(i))
        assert abs(sw.mean() - 2.0) < 1e-10

    def test_sliding(self):
        from dravex.core.buffer import SlidingWindow
        sw = SlidingWindow(3)
        for i in range(10):
            sw.push(float(i))
        assert sw.count == 3
        assert abs(sw.mean() - 8.0) < 1e-10

    def test_variance(self):
        from dravex.core.buffer import SlidingWindow
        sw = SlidingWindow(100)
        for i in range(100):
            sw.push(1.0)
        assert abs(sw.variance()) < 1e-10

    def test_rms(self):
        from dravex.core.buffer import SlidingWindow
        sw = SlidingWindow(4)
        sw.push(3.0)
        sw.push(4.0)
        import math
        assert abs(sw.rms() - math.sqrt(12.5)) < 1e-10
