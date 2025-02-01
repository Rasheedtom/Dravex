"""Tests for pitch detection."""
import math
from dravex.spectral.pitch import autocorrelation_pitch, yin_pitch

class TestPitch:
    def test_sine_pitch(self):
        freq = 440
        sr = 44100
        x = [math.sin(2*math.pi*freq*i/sr) for i in range(4096)]
        detected = autocorrelation_pitch(x, sr)
        assert detected > 0

    def test_yin(self):
        freq = 200
        sr = 44100
        x = [math.sin(2*math.pi*freq*i/sr) for i in range(4096)]
        detected = yin_pitch(x, sr)
        assert detected > 0

    def test_silence(self):
        x = [0.0] * 1024
        detected = autocorrelation_pitch(x, 44100)
        assert detected == 0.0
