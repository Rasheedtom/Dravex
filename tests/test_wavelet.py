"""Tests for wavelet transform."""
from dravex.transform.wavelet import haar_forward, haar_inverse, wavelet_denoise

class TestWavelet:
    def test_haar_roundtrip(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        approx, details = haar_forward(x)
        y = haar_inverse(approx, details)
        for a, b in zip(x, y):
            assert abs(a - b) < 1e-8

    def test_haar_length(self):
        x = [1, 2, 3, 4]
        approx, details = haar_forward(x)
        assert len(approx) == 1

    def test_denoise(self):
        x = [0, 0, 0, 10, 0, 0, 0, 0]
        d = wavelet_denoise(x, threshold=1.0)
        assert len(d) == len(x)
