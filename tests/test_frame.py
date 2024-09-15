"""Tests for frame extraction."""
from dravex.core.frame import Frame, extract_frames
from dravex.core.signal import Signal

class TestFrame:
    def test_create_frame(self):
        f = Frame([1, 2, 3, 4])
        assert f.size == 4

    def test_energy(self):
        f = Frame([1, 1, 1, 1])
        assert abs(f.energy() - 4.0) < 1e-10

    def test_rms(self):
        f = Frame([3.0, 4.0])
        expected = (9 + 16) / 2
        assert abs(f.rms()**2 - expected) < 1e-8

    def test_zero_crossings(self):
        f = Frame([1, -1, 1, -1, 1])
        assert f.zero_crossings() == 4

    def test_windowed(self):
        w = [0.5, 1.0, 0.5]
        f = Frame([2, 4, 6], window=w)
        assert f.windowed_samples == [1.0, 4.0, 3.0]

    def test_extract_frames(self):
        s = Signal(list(range(20)), 44100)
        frames = extract_frames(s, 10, 5)
        assert len(frames) == 3

    def test_autocorrelation(self):
        f = Frame([1, 0, 1, 0])
        ac = f.autocorrelation()
        assert ac[0] > ac[1]
