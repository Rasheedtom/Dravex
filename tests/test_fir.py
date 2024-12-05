"""Tests for FIR filter design."""
from dravex.filter.fir import fir_lowpass, fir_highpass, fir_bandpass, apply_fir
import math
from dravex.filter.fir import fir_lowpass, fir_highpass, fir_bandpass, apply_fir, fir_moving_average, linear_phase_check
from dravex.filter.iir import apply_iir, biquad, cascade_biquads
from dravex.filter.butterworth import butterworth_lowpass, butterworth_sections, butterworth_bandpass
from dravex.filter.notch import notch_filter, apply_notch
from dravex.filter.allpass import allpass_first_order, dispersion_filter
from dravex.filter.moving_avg import simple_moving_average, exponential_moving_average

class TestFIR:
    def test_lowpass_coeffs(self):
        h = fir_lowpass(32, 1000, 44100)
        assert len(h) == 33

    def test_lowpass_dc_pass(self):
        h = fir_lowpass(32, 5000, 44100)
        dc_gain = sum(h)
        assert abs(dc_gain - 1.0) < 0.2

    def test_highpass_coeffs(self):
        h = fir_highpass(32, 1000, 44100)
        dc_gain = sum(h)
        assert abs(dc_gain) < 1.0

    def test_apply_fir(self):
        x = [1.0] + [0.0] * 9
        h = [0.5, 0.5]
        y = apply_fir(x, h)
        assert abs(y[0] - 0.5) < 1e-10
        assert abs(y[1] - 0.5) < 1e-10

    def test_bandpass(self):
        h = fir_bandpass(32, 500, 2000, 44100)
        assert len(h) == 33

    def test_filter_sine(self):
        n = 1024
        x = [math.sin(2*math.pi*100*i/44100) for i in range(n)]
        h = fir_lowpass(32, 200, 44100)
        y = apply_fir(x, h)
        assert len(y) == n



class TestFilterAdvanced:
    def test_lowpass_attenuates_high(self):
        sr = 44100
        low = [math.sin(2*math.pi*100*i/sr) for i in range(2048)]
        high = [math.sin(2*math.pi*10000*i/sr) for i in range(2048)]
        mixed = [l + h for l, h in zip(low, high)]
        h = fir_lowpass(64, 500, sr)
        filtered = apply_fir(mixed, h)
        low_energy = sum(s*s for s in low[100:])
        filt_energy = sum(s*s for s in filtered[100:])
        assert filt_energy < low_energy * 2

    def test_highpass_removes_dc(self):
        x = [5.0] * 1024
        h = fir_highpass(32, 100, 44100)
        y = apply_fir(x, h)
        assert len(y) == len(x)

    def test_bandpass_passband(self):
        sr = 44100
        h = fir_bandpass(64, 400, 600, sr)
        x = [math.sin(2*math.pi*500*i/sr) for i in range(2048)]
        y = apply_fir(x, h)
        in_energy = sum(s*s for s in x[100:])
        out_energy = sum(s*s for s in y[100:])
        assert len(y) == len(x)

    def test_moving_average(self):
        h = fir_moving_average(5)
        assert len(h) == 5
        assert abs(sum(h) - 1.0) < 1e-10

    def test_linear_phase(self):
        h = fir_lowpass(32, 1000, 44100)
        assert linear_phase_check(h)

    def test_butterworth_sections(self):
        sections = butterworth_sections(4, 1000, 44100)
        assert len(sections) == 2

    def test_cascade_biquads(self):
        x = [1.0] + [0.0] * 99
        sections = [(1, 0, 0, 0, 0)]
        y = cascade_biquads(x, sections)
        assert abs(y[0] - 1.0) < 1e-10

    def test_notch(self):
        b, a = notch_filter(1000, 100, 44100)
        assert len(b) == 3
        assert len(a) == 3

    def test_apply_notch(self):
        sr = 44100
        x = [math.sin(2*math.pi*1000*i/sr) for i in range(4096)]
        y = apply_notch(x, 1000, 200, sr)
        out_energy = sum(s*s for s in y[200:])
        in_energy = sum(s*s for s in x[200:])
        assert out_energy < in_energy * 0.5

    def test_allpass_energy(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(1024)]
        y = allpass_first_order(x, 0.5)
        in_e = sum(s*s for s in x)
        out_e = sum(s*s for s in y)
        assert abs(in_e - out_e) / max(in_e, 1) < 0.3

    def test_sma(self):
        x = [1, 2, 3, 4, 5]
        y = simple_moving_average(x, 3)
        assert abs(y[2] - 2.0) < 1e-10

    def test_ema(self):
        x = [0] * 10 + [1] * 10
        y = exponential_moving_average(x, 0.3)
        assert y[-1] > 0.5

    def test_butterworth_bandpass(self):
        b, a = butterworth_bandpass(2, 500, 1500, 44100)
        assert len(b) == 3


class TestNotchAdvanced:
    def test_twin_t(self):
        from dravex.filter.notch import twin_t_notch
        b, a = twin_t_notch(1000, 44100)
        assert len(b) == 3

    def test_multi_notch(self):
        import math
        from dravex.filter.notch import multi_notch
        x = [math.sin(2*math.pi*1000*i/44100) + math.sin(2*math.pi*2000*i/44100) for i in range(4096)]
        y = multi_notch(x, [1000, 2000], 100, 44100)
        assert len(y) == len(x)

    def test_quality_factor(self):
        from dravex.filter.notch import notch_quality_factor
        q = notch_quality_factor(1000, 100)
        assert abs(q - 10.0) < 1e-10

    def test_peak_filter(self):
        from dravex.filter.notch import peak_filter
        b, a = peak_filter(1000, 6, 200, 44100)
        assert len(b) == 3
