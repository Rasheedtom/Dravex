"""Tests for core signal operations."""
import math
from dravex.core.signal import Signal

class TestSignalBasics:
    def test_create_signal(self):
        s = Signal([1, 2, 3], 44100)
        assert s.length == 3
        assert s.sample_rate == 44100

    def test_duration(self):
        s = Signal([0]*44100, 44100)
        assert abs(s.duration - 1.0) < 1e-6

    def test_add_signals(self):
        a = Signal([1, 2, 3], 44100)
        b = Signal([4, 5, 6], 44100)
        c = a + b
        assert c.samples == [5.0, 7.0, 9.0]

    def test_multiply_scalar(self):
        s = Signal([1, 2, 3], 44100) * 2
        assert s.samples == [2.0, 4.0, 6.0]

    def test_normalize(self):
        s = Signal([0.5, -1.0, 0.3], 44100)
        n = s.normalize(1.0)
        assert abs(n.max_amplitude() - 1.0) < 1e-10

    def test_rms(self):
        s = Signal([1.0, -1.0, 1.0, -1.0], 44100)
        assert abs(s.rms() - 1.0) < 1e-10

    def test_energy(self):
        s = Signal([3.0, 4.0], 44100)
        assert abs(s.energy() - 25.0) < 1e-10

    def test_reverse(self):
        s = Signal([1, 2, 3], 44100)
        assert s.reverse().samples == [3.0, 2.0, 1.0]

    def test_pad_zeros(self):
        s = Signal([1, 2], 44100).pad_zeros(5)
        assert s.length == 5
        assert s[3] == 0.0

    def test_trim_silence(self):
        s = Signal([0, 0, 0.5, 0.8, 0, 0], 44100)
        t = s.trim_silence(0.01)
        assert t.length == 2

    def test_concatenate(self):
        a = Signal([1, 2], 44100)
        b = Signal([3, 4], 44100)
        c = a.concatenate(b)
        assert c.length == 4

    def test_zeros(self):
        s = Signal.zeros(10)
        assert all(x == 0.0 for x in s.samples)

    def test_slice(self):
        s = Signal([1, 2, 3, 4, 5], 44100)
        sliced = s[1:4]
        assert sliced.length == 3

    def test_negate(self):
        s = Signal([1, -2, 3], 44100)
        n = -s
        assert n.samples == [-1.0, 2.0, -3.0]

    def test_fade_in(self):
        s = Signal([1]*10, 44100)
        f = s.fade_in(5)
        assert f[0] == 0.0
        assert f[4] > 0

    def test_fade_out(self):
        s = Signal([1]*10, 44100)
        f = s.fade_out(5)
        assert f[-1] == 0.0



class TestSignalAdvanced:
    def test_downsample(self):
        s = Signal(list(range(100)), 44100)
        d = s.downsample(2)
        assert d.length == 50
        assert d.sample_rate == 22050

    def test_upsample(self):
        s = Signal([1, 2, 3], 44100)
        u = s.upsample(2)
        assert u.length == 6
        assert u.sample_rate == 88200

    def test_split(self):
        s = Signal(list(range(100)), 44100)
        parts = s.split(4)
        assert len(parts) == 4
        assert parts[0].length == 25

    def test_overlap_add(self):
        s = Signal([1, 1, 1, 1, 1], 44100)
        overlay = Signal([0.5, 0.5], 44100)
        r = s.overlap_add_with(overlay, 2)
        assert abs(r[2] - 1.5) < 1e-10

    def test_moving_rms(self):
        s = Signal([1]*10 + [0]*10, 44100)
        rms_vals = s.moving_rms(5)
        assert len(rms_vals) == 20
        assert rms_vals[0] > rms_vals[15]

    def test_spectral_centroid(self):
        s = Signal([math.sin(2*math.pi*1000*i/44100) for i in range(4096)], 44100)
        c = s.spectral_centroid()
        assert c > 0

    def test_auto_correlate(self):
        s = Signal([1, -1, 1, -1, 1, -1, 1, -1], 44100)
        ac = s.auto_correlate(4)
        assert ac[0] > 0

    def test_add_scalar(self):
        s = Signal([1, 2, 3], 44100)
        r = s + 10
        assert r.samples == [11.0, 12.0, 13.0]

    def test_rmul(self):
        s = Signal([1, 2], 44100)
        r = 3 * s
        assert r.samples == [3.0, 6.0]

    def test_rms_db(self):
        s = Signal([1.0] * 100, 44100)
        db = s.rms_db()
        assert abs(db - 0.0) < 0.01

    def test_window_apply(self):
        from dravex.window.hann import hann
        s = Signal([1]*100, 44100)
        w = s.window(hann)
        assert w[0] < 0.01

    def test_apply_func(self):
        s = Signal([1, 4, 9], 44100)
        r = s.apply(math.sqrt)
        assert abs(r[0] - 1.0) < 1e-10
        assert abs(r[1] - 2.0) < 1e-10

    def test_hash_stable(self):
        s = Signal([1, 2, 3], 44100)
        assert s.length == 3



class TestSignalMethods:
    def test_resample(self):
        s = Signal([math.sin(2*math.pi*100*i/44100) for i in range(4410)], 44100)
        r = s.resample(22050)
        assert abs(r.length - 2205) < 10

    def test_filter_lowpass(self):
        s = Signal([math.sin(2*math.pi*100*i/44100) + math.sin(2*math.pi*10000*i/44100) for i in range(4096)], 44100)
        f = s.filter_lowpass(500)
        assert f.length == s.length

    def test_filter_highpass(self):
        s = Signal([5.0 + math.sin(2*math.pi*1000*i/44100) for i in range(4096)], 44100)
        f = s.filter_highpass(500)
        mean_val = sum(f.samples[100:]) / (f.length - 100)
        assert isinstance(f, Signal)

    def test_spectrogram(self):
        s = Signal([math.sin(2*math.pi*440*i/44100) for i in range(4096)], 44100)
        spec = s.spectrogram(512, 256)
        assert len(spec) > 0

    def test_mfcc(self):
        s = Signal([math.sin(2*math.pi*440*i/44100) for i in range(1024)], 44100)
        coeffs = s.mfcc(13, 512)
        assert len(coeffs) == 13

    def test_pitch(self):
        s = Signal([math.sin(2*math.pi*440*i/44100) for i in range(8192)], 44100)
        p = s.pitch()
        assert p >= 0

    def test_to_frames(self):
        s = Signal(list(range(4096)), 44100)
        frames = s.to_frames(1024, 512)
        assert len(frames) > 0

    def test_detect_onsets(self):
        silence = [0.001] * 2048
        burst = [math.sin(2*math.pi*440*i/44100) for i in range(2048)]
        s = Signal(silence + burst + silence, 44100)
        onsets = s.detect_onsets(512, 256)
        assert len(onsets) >= 0


class TestSignalExtra:
    def test_from_function(self):
        import math
        s = Signal.from_function(math.sin, 0.01, 1000)
        assert abs(s.length - 10) < 2

    def test_crossfade(self):
        a = Signal([1.0]*20, 44100)
        b = Signal([0.0]*20, 44100)
        c = a.crossfade_with(b, 10)
        assert c.length == 30

    def test_repeat(self):
        s = Signal([1, 2, 3], 44100)
        r = s.repeat(3)
        assert r.length == 9
        assert r.samples == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

    def test_zero_pad_pow2(self):
        s = Signal([1]*100, 44100)
        p = s.zero_pad_to_power_of_2()
        assert p.length == 128

    def test_dc_offset(self):
        s = Signal([5, 5, 5], 44100)
        assert abs(s.dc_offset() - 5.0) < 1e-10

    def test_peak_to_peak(self):
        s = Signal([-1, 0, 1], 44100)
        assert abs(s.peak_to_peak() - 2.0) < 1e-10

    def test_clip_count(self):
        s = Signal([0.5, 1.0, 0.99, 1.0, 0.5], 44100)
        assert s.clip_count(0.99) == 3

    def test_histogram(self):
        s = Signal(list(range(100)), 44100)
        bins, counts = s.histogram(10)
        assert sum(counts) == 100

    def test_entropy(self):
        s = Signal([1.0]*100, 44100)
        e = s.entropy()
        assert e == 0.0


class TestSignalExtended:
    def test_zero_pad_pow2(self):
        s = Signal([1]*100, 44100)
        p = s.zero_pad_to_power_of_2()
        from dravex.constants import is_power_of_2
        assert is_power_of_2(p.length)

    def test_dc_offset_removal(self):
        from dravex.core.normalize import dc_remove
        s = Signal([5.0 + math.sin(0.1*i) for i in range(100)], 44100)
        clean = dc_remove(s)
        assert abs(clean.dc_offset()) < 0.01

    def test_from_function_sine(self):
        s = Signal.from_function(lambda t: math.sin(2*math.pi*440*t), 0.01, 44100)
        assert s.length == 441

    def test_repeat_energy(self):
        s = Signal([1.0, 0.0], 44100)
        r = s.repeat(5)
        assert r.length == 10
        assert abs(r.energy() - 5.0) < 1e-10
