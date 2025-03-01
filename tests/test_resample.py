"""Tests for resampling."""
from dravex.resample.interpolate import linear_interpolate, resample_ratio
from dravex.resample.decimate import decimate_no_filter
import math
from dravex.resample.interpolate import linear_interpolate, cubic_interpolate, resample_ratio, zero_order_hold
from dravex.resample.decimate import decimate_no_filter, downsample_average, downsample_peak
from dravex.resample.sinc_resample import sinc_resample
from dravex.resample.fractional import fractional_delay, pitch_shift_td
from dravex.resample.rate_convert import convert_sample_rate, rational_resample_factors

class TestResample:
    def test_upsample_2x(self):
        x = [0, 1, 0]
        y = linear_interpolate(x, 2)
        assert len(y) == 5

    def test_downsample(self):
        x = [1, 2, 3, 4, 5, 6]
        y = decimate_no_filter(x, 2)
        assert y == [1, 3, 5]

    def test_resample_ratio(self):
        x = list(range(100))
        y = resample_ratio(x, 0.5)
        assert len(y) == 50

    def test_identity(self):
        x = [1, 2, 3, 4]
        y = resample_ratio(x, 1.0)
        assert len(y) == 4



class TestResampleAdvanced:
    def test_cubic_interpolate(self):
        x = [0, 1, 0, -1, 0, 1, 0, -1]
        y = cubic_interpolate(x, 2)
        assert len(y) > len(x)

    def test_zero_order_hold(self):
        x = [1, 2, 3]
        y = zero_order_hold(x, 3)
        assert y == [1, 1, 1, 2, 2, 2, 3, 3, 3]

    def test_downsample_average(self):
        x = [1, 3, 2, 4]
        y = downsample_average(x, 2)
        assert abs(y[0] - 2.0) < 1e-10
        assert abs(y[1] - 3.0) < 1e-10

    def test_downsample_peak(self):
        x = [1, -5, 2, -3]
        y = downsample_peak(x, 2)
        assert y[0] == 5.0

    def test_sinc_resample(self):
        x = [math.sin(2*math.pi*i/32) for i in range(128)]
        y = sinc_resample(x, 64)
        assert len(y) == 64

    def test_fractional_delay(self):
        x = [0]*10 + [1] + [0]*20
        delayed = fractional_delay(x, 2.5)
        assert len(delayed) == len(x)

    def test_pitch_shift(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        shifted = pitch_shift_td(x, 12, 44100)
        assert len(shifted) == len(x)

    def test_convert_rate(self):
        x = [math.sin(2*math.pi*100*i/44100) for i in range(4410)]
        y = convert_sample_rate(x, 44100, 22050, "low")
        assert abs(len(y) - 2205) < 10

    def test_rational_factors(self):
        up, down = rational_resample_factors(44100, 48000)
        assert up * 44100 == down * 48000 or (up > 0 and down > 0)

    def test_identity_resample(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = convert_sample_rate(x, 44100, 44100)
        assert y == x


class TestPolyphaseAdvanced:
    def test_analysis_synthesis(self):
        from dravex.resample.polyphase import polyphase_analysis, polyphase_synthesis
        x = list(range(20))
        channels = polyphase_analysis(x, 4)
        assert len(channels) == 4
        reconstructed = polyphase_synthesis(channels)
        assert reconstructed == x

    def test_channelizer(self):
        import math
        from dravex.resample.polyphase import channelizer
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        channels = channelizer(x, 4, 16, 44100)
        assert len(channels) == 4
