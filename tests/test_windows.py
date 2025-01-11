"""Tests for window functions."""
import math
from dravex.window.hamming import hamming
from dravex.window.hann import hann
from dravex.window.blackman import blackman
from dravex.window.kaiser import kaiser
from dravex.window.gaussian import gaussian_window
from dravex.window.tukey import tukey
from dravex.window.hamming import hamming, hamming_periodic, hamming_bandwidth
from dravex.window.hann import hann, hann_poisson, cosine_sum_window
from dravex.window.blackman import blackman, blackman_harris, blackman_nuttall, flat_top
from dravex.window.kaiser import kaiser, kaiser_beta_from_atten, kaiser_order
from dravex.window.gaussian import gaussian_window, confined_gaussian, gabor_window
from dravex.window.tukey import tukey, planck_taper, cosine_window, welch_window

class TestWindows:
    def test_hamming_length(self):
        assert len(hamming(10)) == 10

    def test_hamming_symmetric(self):
        w = hamming(11)
        assert abs(w[0] - w[-1]) < 1e-10

    def test_hann_endpoints(self):
        w = hann(10)
        assert abs(w[0]) < 1e-10

    def test_hann_peak(self):
        w = hann(11)
        assert abs(w[5] - 1.0) < 1e-10

    def test_blackman_length(self):
        assert len(blackman(20)) == 20

    def test_kaiser_beta_zero(self):
        w = kaiser(10, beta=0)
        for v in w:
            assert abs(v - 1.0) < 0.01

    def test_gaussian_peak(self):
        w = gaussian_window(11, 0.4)
        assert abs(w[5] - 1.0) < 1e-10

    def test_tukey_rectangular(self):
        w = tukey(10, alpha=0)
        assert all(abs(v - 1.0) < 1e-10 for v in w)

    def test_tukey_hann(self):
        w_tukey = tukey(11, alpha=1.0)
        w_hann = hann(11)
        for a, b in zip(w_tukey, w_hann):
            assert abs(a - b) < 0.01

    def test_all_positive(self):
        for func in [hamming, hann, blackman]:
            w = func(20)
            assert all(v >= -1e-10 for v in w)



class TestWindowAdvanced:
    def test_hamming_periodic(self):
        w = hamming_periodic(10)
        assert len(w) == 10

    def test_hamming_bandwidth_calc(self):
        bw = hamming_bandwidth(1024, 44100)
        assert bw > 0

    def test_hann_poisson(self):
        w = hann_poisson(64, alpha=2.0)
        assert len(w) == 64
        assert all(v >= 0 for v in w)

    def test_cosine_sum(self):
        w = cosine_sum_window(64, [0.5, 0.5])
        assert len(w) == 64

    def test_blackman_harris(self):
        w = blackman_harris(128)
        assert len(w) == 128
        assert all(v >= -0.01 for v in w)

    def test_blackman_nuttall(self):
        w = blackman_nuttall(128)
        assert len(w) == 128

    def test_flat_top(self):
        w = flat_top(64)
        assert len(w) == 64

    def test_kaiser_beta(self):
        beta = kaiser_beta_from_atten(60)
        assert beta > 0

    def test_kaiser_order(self):
        order = kaiser_order(500, 50, 44100)
        assert order > 0

    def test_confined_gaussian(self):
        w = confined_gaussian(64, 0.4)
        assert abs(w[0]) < 0.01

    def test_gabor(self):
        w = gabor_window(64, 0.4, 0.1)
        assert len(w) == 64

    def test_planck_taper(self):
        w = planck_taper(64, 0.1)
        assert len(w) == 64
        assert abs(w[32] - 1.0) < 0.01

    def test_cosine_window(self):
        w = cosine_window(64)
        assert len(w) == 64
        assert abs(w[0]) < 1e-10

    def test_welch(self):
        w = welch_window(64)
        assert len(w) == 64
        assert abs(w[32] - 1.0) < 0.01

    def test_window_energy_normalization(self):
        for func in [hamming, hann, blackman]:
            w = func(1024)
            energy = sum(v*v for v in w)
            assert energy > 0

    def test_single_sample_windows(self):
        for func in [hamming, hann, blackman, kaiser, gaussian_window, tukey]:
            w = func(1)
            assert len(w) == 1
            assert abs(w[0] - 1.0) < 1e-10
