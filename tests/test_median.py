"""Tests for median filter."""
from dravex.filter.median import median_filter, adaptive_median

class TestMedianFilter:
    def test_basic(self):
        x = [1, 1, 100, 1, 1]
        y = median_filter(x, 3)
        assert y[2] == 1.0

    def test_preserves_monotone(self):
        x = [1, 2, 3, 4, 5]
        y = median_filter(x, 3)
        for i in range(1, len(y)-1):
            assert y[i] == x[i]

    def test_adaptive(self):
        x = [1, 1, 100, 1, 1, 1, 1]
        y = adaptive_median(x, 3, 7)
        assert y[2] < 50
