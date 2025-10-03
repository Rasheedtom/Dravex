"""Tests for signal segmentation and normalization."""
import math
from dravex.core.signal import Signal
from dravex.core.segment import segment_fixed, segment_by_silence, segment_by_energy, concatenate_signals
from dravex.core.normalize import peak_normalize, rms_normalize, dc_remove, pre_emphasis, de_emphasis, clip, soft_clip

class TestSegment:
    def test_fixed_segments(self):
        s = Signal(list(range(100)), 44100)
        segs = segment_fixed(s, 25)
        assert len(segs) == 4

    def test_fixed_overlap(self):
        s = Signal(list(range(100)), 44100)
        segs = segment_fixed(s, 25, 10)
        assert len(segs) > 4

    def test_by_silence(self):
        data = [0.001]*50 + [0.5]*100 + [0.001]*50 + [0.8]*100
        s = Signal(data, 44100)
        segs = segment_by_silence(s, 0.01, 20)
        assert len(segs) >= 2

    def test_by_energy(self):
        data = [0.001]*1024 + [0.5]*2048 + [0.001]*1024
        s = Signal(data, 44100)
        segs = segment_by_energy(s, 512, 256, -30)
        assert len(segs) >= 1

    def test_concatenate(self):
        a = Signal([1, 2, 3], 44100)
        b = Signal([4, 5, 6], 44100)
        c = concatenate_signals([a, b])
        assert c.length == 6

    def test_concatenate_crossfade(self):
        a = Signal([1.0]*10, 44100)
        b = Signal([0.0]*10, 44100)
        c = concatenate_signals([a, b], crossfade=5)
        assert len(c.samples) < 20

class TestNormalize:
    def test_peak(self):
        s = Signal([0.5, -0.3], 44100)
        n = peak_normalize(s, 1.0)
        assert abs(n.max_amplitude() - 1.0) < 1e-10

    def test_rms(self):
        s = Signal([1.0]*100, 44100)
        n = rms_normalize(s, 0.5)
        assert abs(n.rms() - 0.5) < 1e-6

    def test_dc_remove(self):
        s = Signal([5.0, 5.0, 5.0], 44100)
        dc = dc_remove(s)
        assert abs(sum(dc.samples)) < 1e-10

    def test_pre_emphasis(self):
        s = Signal([1, 2, 3, 4], 44100)
        pe = pre_emphasis(s, 0.97)
        assert pe.length == 4

    def test_de_emphasis_inverse(self):
        s = Signal([1, 0, 0, 0, 0]*20, 44100)
        pe = pre_emphasis(s, 0.97)
        de = de_emphasis(pe, 0.97)
        for a, b in zip(s.samples[:50], de.samples[:50]):
            assert abs(a - b) < 0.5

    def test_clip(self):
        s = Signal([2.0, -2.0, 0.5], 44100)
        c = clip(s, -1, 1)
        assert c[0] == 1.0
        assert c[1] == -1.0
        assert c[2] == 0.5

    def test_soft_clip(self):
        s = Signal([0.5, 2.0, -2.0], 44100)
        sc = soft_clip(s, 0.8)
        assert abs(sc[0]) < 0.8
        assert abs(sc[1]) < 0.8
