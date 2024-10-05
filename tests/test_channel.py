"""Tests for multi-channel signals."""
from dravex.core.channel import MultiChannel
from dravex.core.signal import Signal

class TestMultiChannel:
    def test_stereo(self):
        mc = MultiChannel([[1, 2, 3], [4, 5, 6]])
        assert mc.n_channels == 2
        assert mc.length == 3

    def test_to_mono(self):
        mc = MultiChannel([[1, 1], [3, 3]])
        mono = mc.to_mono()
        assert abs(mono[0] - 2.0) < 1e-10

    def test_from_interleaved(self):
        mc = MultiChannel.from_interleaved([1, 4, 2, 5, 3, 6], 2)
        assert mc.channel(0).samples == [1.0, 2.0, 3.0]
        assert mc.channel(1).samples == [4.0, 5.0, 6.0]

    def test_to_interleaved(self):
        mc = MultiChannel([[1, 2], [3, 4]])
        il = mc.to_interleaved()
        assert il == [1.0, 3.0, 2.0, 4.0]

    def test_apply_gain(self):
        mc = MultiChannel([[1, 1], [1, 1]])
        gained = mc.apply_gain([2.0, 0.5])
        assert gained.channel(0)[0] == 2.0
        assert gained.channel(1)[0] == 0.5


class TestMultiChannelAdvanced:
    def test_mid_side(self):
        mc = MultiChannel([[1, 1, 1], [0, 0, 0]])
        ms = mc.mid_side()
        assert abs(ms.channel(0)[0] - 0.5) < 1e-10

    def test_mid_side_roundtrip(self):
        mc = MultiChannel([[1, 2, 3], [4, 5, 6]])
        ms = mc.mid_side()
        lr = ms.from_mid_side()
        for a, b in zip(mc.channel(0).samples, lr.channel(0).samples):
            assert abs(a - b) < 1e-10

    def test_correlation_identical(self):
        mc = MultiChannel([[1, 2, 3], [1, 2, 3]])
        assert abs(mc.correlation() - 1.0) < 1e-6

    def test_peak_normalize(self):
        mc = MultiChannel([[0.5, -0.5], [0.3, -0.3]])
        normalized = mc.peak_normalize_all(1.0)
        peak = max(normalized.channel(0).max_amplitude(), normalized.channel(1).max_amplitude())
        assert abs(peak - 1.0) < 1e-10

    def test_rms_per_channel(self):
        mc = MultiChannel([[1, 1, 1], [2, 2, 2]])
        rms_vals = mc.rms_per_channel()
        assert abs(rms_vals[0] - 1.0) < 1e-10
        assert abs(rms_vals[1] - 2.0) < 1e-10

    def test_trim_to_shortest(self):
        mc = MultiChannel([[1, 2, 3, 4], [5, 6]])
        trimmed = mc.trim_to_shortest()
        assert trimmed.length == 2
