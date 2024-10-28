"""Tests for FFT."""
import math
from dravex.transform.fft import fft, ifft, rfft, magnitude_phase
from dravex.transform.fft import fft, ifft, rfft, fft_shift, power_spectrum, cross_spectrum, spectral_flatness

class TestFFT:
    def test_impulse(self):
        result = fft([1, 0, 0, 0])
        assert all(abs(re - 1.0) < 1e-10 for re, im in result)

    def test_dc(self):
        result = fft([1, 1, 1, 1])
        assert abs(result[0][0] - 4.0) < 1e-10
        assert abs(result[1][0]) < 1e-10

    def test_inverse(self):
        original = [1.0, 2.0, 3.0, 4.0]
        X = fft(original)
        x = ifft(X)
        for orig, (re, im) in zip(original, x):
            assert abs(orig - re) < 1e-8

    def test_rfft_length(self):
        mags = rfft([1, 0, 0, 0, 0, 0, 0, 0])
        assert len(mags) == 5

    def test_parseval(self):
        x = [1, 2, 3, 4]
        time_energy = sum(s*s for s in x)
        X = fft(x)
        freq_energy = sum(re*re + im*im for re, im in X) / len(X)
        assert abs(time_energy - freq_energy) < 1e-6

    def test_sine(self):
        n = 64
        x = [math.sin(2 * math.pi * 4 * i / n) for i in range(n)]
        mags = rfft(x)
        peak = max(range(len(mags)), key=lambda i: mags[i])
        assert peak == 4

    def test_magnitude_phase(self):
        X = [(3, 4), (1, 0)]
        mags, phases = magnitude_phase(X)
        assert abs(mags[0] - 5.0) < 1e-10
        assert abs(mags[1] - 1.0) < 1e-10



class TestFFTAdvanced:
    def test_fft_shift(self):
        x = [1, 2, 3, 4]
        shifted = fft_shift(x)
        assert shifted == [3, 4, 1, 2]

    def test_power_spectrum(self):
        x = [1, 0, 0, 0]
        ps = power_spectrum(x)
        assert all(abs(p - 1.0) < 1e-8 for p in ps)

    def test_cross_spectrum(self):
        x = [1, 0, 0, 0]
        y = [0, 1, 0, 0]
        cs = cross_spectrum(x, y)
        assert len(cs) == 4

    def test_spectral_flatness_white(self):
        mags = [1.0] * 100
        sf = spectral_flatness(mags)
        assert abs(sf - 1.0) < 0.01

    def test_spectral_flatness_tonal(self):
        mags = [0.0] * 100
        mags[10] = 100.0
        sf = spectral_flatness(mags)
        assert sf <= 1.01

    def test_large_fft(self):
        n = 1024
        x = [math.sin(2*math.pi*100*i/n) for i in range(n)]
        mags = rfft(x)
        peak = max(range(len(mags)), key=lambda i: mags[i])
        assert 98 <= peak <= 102

    def test_linearity(self):
        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        fa = fft(a)
        fb = fft(b)
        combined = fft([ai + bi for ai, bi in zip(a, b)])
        for (r1, i1), (r2, i2), (r3, i3) in zip(fa, fb, combined):
            assert abs((r1 + r2) - r3) < 1e-8
            assert abs((i1 + i2) - i3) < 1e-8

    def test_conjugate_symmetry(self):
        x = [1, 2, 3, 4]
        X = fft(x)
        n = len(X)
        for k in range(1, n // 2):
            assert abs(X[k][0] - X[n-k][0]) < 1e-8
            assert abs(X[k][1] + X[n-k][1]) < 1e-8

    def test_multiple_roundtrips(self):
        x = [0.5, -0.3, 0.8, -0.1]
        for _ in range(5):
            X = fft(x)
            result = ifft(X)
            x = [re for re, im in result]
        for orig, final in zip([0.5, -0.3, 0.8, -0.1], x):
            assert abs(orig - final) < 1e-6
