"""Tests for MFCC."""
from dravex.spectral.mfcc import mfcc, mel_filterbank, delta_mfcc
import math

class TestMFCC:
    def test_mfcc_length(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(512)]
        coeffs = mfcc(x, 44100, 13, 26, 512)
        assert len(coeffs) == 13

    def test_filterbank(self):
        fb = mel_filterbank(26, 512, 44100)
        assert len(fb) == 26
        assert len(fb[0]) == 257

    def test_delta(self):
        frames = [[float(i)] * 13 for i in range(10)]
        d = delta_mfcc(frames)
        assert len(d) == 10
