"""Tests for correlation."""
from dravex.correlation.autocorr import autocorrelation, normalized_autocorrelation
from dravex.correlation.crosscorr import cross_correlation, similarity
import math
from dravex.correlation.autocorr import autocorrelation, normalized_autocorrelation, partial_autocorrelation
from dravex.correlation.crosscorr import cross_correlation, normalized_cross_correlation, similarity, pearson_correlation, sliding_correlation
from dravex.correlation.coherence import coherence
from dravex.correlation.delay import gcc_phat, onset_delay, peak_delay
from dravex.correlation.matched import matched_filter, detect_template, snr_gain

class TestCorrelation:
    def test_autocorr_peak(self):
        x = [1, 0, -1, 0] * 10
        r = autocorrelation(x, 20)
        assert r[0] >= max(r[1:])

    def test_normalized(self):
        x = [1, 2, 3, 4, 5]
        r = normalized_autocorrelation(x, 5)
        assert abs(r[0] - 1.0) < 1e-10

    def test_cross_corr(self):
        x = [0, 0, 1, 0, 0]
        y = [0, 0, 0, 1, 0]
        r = cross_correlation(x, y)
        assert max(r) > 0

    def test_similarity_identical(self):
        x = [1, 2, 3, 4, 5]
        assert abs(similarity(x, x) - 1.0) < 1e-10

    def test_similarity_orthogonal(self):
        x = [1, 0, 0]
        y = [0, 1, 0]
        assert abs(similarity(x, y)) < 1e-10



class TestCorrelationAdvanced:
    def test_autocorr_periodic(self):
        period = 20
        x = [math.sin(2*math.pi*i/period) for i in range(200)]
        r = normalized_autocorrelation(x, 50)
        assert abs(r[period] - 1.0) < 0.1

    def test_partial_autocorrelation(self):
        x = [math.sin(0.1*i) + 0.5*math.sin(0.3*i) for i in range(500)]
        pacf = partial_autocorrelation(x, 10)
        assert len(pacf) == 10

    def test_cross_corr_delayed(self):
        x = [0]*10 + [1, 2, 3, 2, 1] + [0]*10
        y = [0]*13 + [1, 2, 3, 2, 1] + [0]*7
        r = cross_correlation(x, y, 10)
        peak = max(r)
        assert peak > 0

    def test_normalized_xcorr(self):
        x = [1, 0, -1, 0] * 10
        r = normalized_cross_correlation(x, x)
        assert max(r) > 0.9

    def test_pearson(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert abs(pearson_correlation(x, y) - 1.0) < 1e-10

    def test_pearson_anticorrelated(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert abs(pearson_correlation(x, y) + 1.0) < 1e-10

    def test_sliding_correlation(self):
        x = list(range(100))
        y = list(range(100))
        sc = sliding_correlation(x, y, 20)
        assert all(abs(c - 1.0) < 0.01 for c in sc)

    def test_coherence(self):
        sr = 44100
        x = [math.sin(2*math.pi*440*i/sr) for i in range(4096)]
        y = [math.sin(2*math.pi*440*i/sr + 0.5) for i in range(4096)]
        freqs, coh = coherence(x, y, 512, 256, sr)
        assert len(coh) > 0

    def test_gcc_phat(self):
        x = [0]*100 + [1, 2, 3, 2, 1] + [0]*100
        y = [0]*103 + [1, 2, 3, 2, 1] + [0]*97
        delay, _ = gcc_phat(x, y, 44100)
        assert isinstance(delay, float)

    def test_onset_delay(self):
        x = [0]*10 + [1]*10
        y = [0]*15 + [1]*10
        d = onset_delay(x, y)
        assert d == 5

    def test_matched_filter(self):
        template = [1, 2, 3, 2, 1]
        signal = [0]*20 + template + [0]*20
        mf = matched_filter(signal, template)
        peak_idx = max(range(len(mf)), key=lambda i: mf[i])
        assert 22 < peak_idx < 28

    def test_detect_template(self):
        template = [1, 1, 1]
        signal = [0]*10 + [1, 1, 1] + [0]*10 + [1, 1, 1] + [0]*10
        dets = detect_template(signal, template, 0.5)
        assert len(dets) >= 1

    def test_snr_gain(self):
        g = snr_gain(100)
        assert abs(g - 20.0) < 0.1
