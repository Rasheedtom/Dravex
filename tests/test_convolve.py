"""Tests for convolution."""
from dravex.convolve.direct import convolve, convolve_same, correlate
from dravex.convolve.overlap_add import fft_convolve
import math
from dravex.convolve.direct import convolve, convolve_symmetric, deconvolve_td, running_average
from dravex.convolve.overlap_add import overlap_add, fft_convolve
from dravex.convolve.overlap_save import overlap_save
from dravex.convolve.deconvolve import wiener_deconvolve, inverse_filter
from dravex.convolve.fast_conv import auto_convolve, circular_convolve, autocorrelation_fft

class TestConvolution:
    def test_impulse(self):
        x = [1, 2, 3]
        h = [1, 0, 0]
        y = convolve(x, h)
        assert y[:3] == [1.0, 2.0, 3.0]

    def test_identity(self):
        x = [1, 2, 3, 4]
        h = [1]
        y = convolve(x, h)
        assert y == [1.0, 2.0, 3.0, 4.0]

    def test_length(self):
        x = [1, 2, 3]
        h = [1, 1]
        y = convolve(x, h)
        assert len(y) == 4

    def test_same(self):
        x = [1, 2, 3, 4, 5]
        h = [1, 1, 1]
        y = convolve_same(x, h)
        assert len(y) == 5

    def test_fft_matches_direct(self):
        x = [1, 2, 3, 4, 0, 0, 0, 0]
        h = [0.5, 0.5, 0, 0, 0, 0, 0, 0]
        d = convolve(x, h)
        f = fft_convolve(x, h)
        for a, b in zip(d[:8], f[:8]):
            assert abs(a - b) < 1.0

    def test_correlate(self):
        x = [0, 0, 1, 0, 0]
        h = [1, 0, 0, 0, 0]
        r = correlate(x, h)
        assert max(r) > 0



class TestConvolveAdvanced:
    def test_overlap_add_matches(self):
        x = [float(i) for i in range(256)]
        h = [0.25, 0.5, 0.25]
        direct_result = convolve(x, h)
        ola_result = overlap_add(x, h, 64)
        for i in range(len(direct_result)):
            assert abs(direct_result[i] - ola_result[i]) < 0.01

    def test_overlap_save_matches(self):
        x = [math.sin(0.1*i) for i in range(256)]
        h = [0.5, 0.5]
        direct_result = convolve(x, h)
        ols_result = overlap_save(x, h, 64)
        for i in range(10, min(len(direct_result), len(ols_result)) - 10):
            assert abs(direct_result[i] - ols_result[i]) < 0.1

    def test_wiener_deconv(self):
        x = [1, 0, 0, 0, 0, 0, 0, 0]
        h = [1, 0.5, 0.25, 0, 0, 0, 0, 0]
        y = fft_convolve(x, h)[:8]
        recovered = wiener_deconvolve(y, h, 0.001)
        assert abs(recovered[0] - 1.0) < 0.3

    def test_auto_convolve_small(self):
        x = [1, 2, 3]
        h = [1, 1]
        result = auto_convolve(x, h, threshold=10)
        assert len(result) == 4

    def test_circular(self):
        x = [1, 2, 3, 4]
        h = [1, 0, 0, 0]
        y = circular_convolve(x, h)
        for a, b in zip(x, y):
            assert abs(a - b) < 1.0

    def test_autocorrelation_fft(self):
        x = [1, 0, -1, 0] * 8
        ac = autocorrelation_fft(x)
        assert ac[0] > 0
        assert ac[0] >= max(ac[1:])

    def test_running_average(self):
        x = [0]*5 + [10]*5 + [0]*5
        y = running_average(x, 3)
        assert len(y) == len(x)

    def test_symmetric_kernel(self):
        x = [1, 2, 3, 4, 5]
        h = [0.25, 0.5, 0.25]
        y = convolve_symmetric(x, h)
        y_ref = convolve(x, h)
        for a, b in zip(y[:len(x)], y_ref[:len(x)]):
            assert abs(a - b) < 0.1

    def test_deconvolve_td(self):
        h = [1, 0.5]
        x = [1, 0, 0, 0, 0]
        y = convolve(x, h)
        recovered = deconvolve_td(y, h)
        assert abs(recovered[0] - 1.0) < 0.01


class TestPartitionedConv:
    def test_partitioned_matches_direct(self):
        from dravex.convolve.overlap_save import partitioned_convolution
        from dravex.convolve.direct import convolve
        x = [float(i)/100 for i in range(200)]
        h = [0.25, 0.5, 0.25]
        direct = convolve(x, h)
        partitioned = partitioned_convolution(x, h, 64)
        for i in range(len(direct)):
            if i < len(partitioned):
                assert abs(direct[i] - partitioned[i]) < 1.5
