"""Tests for Butterworth filter."""
from dravex.filter.butterworth import butterworth_lowpass, butterworth_highpass, butterworth_sections

class TestButterworth:
    def test_lowpass_order1(self):
        b, a = butterworth_lowpass(1, 1000, 44100)
        assert len(b) == 2
        assert len(a) == 2

    def test_lowpass_order2(self):
        b, a = butterworth_lowpass(2, 1000, 44100)
        assert len(b) == 3
        assert len(a) == 3

    def test_highpass(self):
        b, a = butterworth_highpass(2, 1000, 44100)
        dc = sum(b)
        assert abs(dc) < 0.1

    def test_sections(self):
        sections = butterworth_sections(4, 1000, 44100)
        assert len(sections) == 2
