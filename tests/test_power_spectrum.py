"""Tests for power spectral density."""
import math
from dravex.spectral.power import periodogram, welch_psd, band_power

class TestPSD:
    def test_periodogram(self):
        x = [math.sin(2*math.pi*1000*i/44100) for i in range(4096)]
        freqs, psd = periodogram(x, 44100)
        assert len(psd) > 0
        peak_idx = max(range(len(psd)), key=lambda i: psd[i])
        assert 900 < freqs[peak_idx] < 1100

    def test_welch(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(8192)]
        freqs, psd = welch_psd(x, 1024, 512, 44100)
        assert len(psd) > 0

    def test_band_power(self):
        freqs = [0, 100, 200, 300, 400]
        psd = [1, 2, 3, 2, 1]
        bp = band_power(psd, freqs, 100, 300)
        assert bp > 0
