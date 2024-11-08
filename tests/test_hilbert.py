"""Tests for Hilbert transform."""
import math
from dravex.transform.hilbert import hilbert_transform, instantaneous_amplitude, analytic_signal

class TestHilbert:
    def test_cosine(self):
        n = 256
        x = [math.cos(2*math.pi*10*i/n) for i in range(n)]
        env = instantaneous_amplitude(x)
        mid = len(env) // 2
        assert abs(env[mid] - 1.0) < 0.15

    def test_analytic(self):
        x = [1, 0, -1, 0] * 16
        a = analytic_signal(x)
        assert len(a) == len(x)

    def test_hilbert_length(self):
        x = [0.0] * 128
        ht = hilbert_transform(x)
        assert len(ht) == 128
