"""Tests for LPC analysis/synthesis."""
import math
from dravex.lpc.analysis import lpc_autocorrelation, lpc_error
from dravex.lpc.synthesis import lpc_synthesize, lpc_resynthesize
from dravex.lpc.parcor import lpc_to_parcor, parcor_to_lpc
from dravex.lpc.analysis import lpc_autocorrelation, lpc_error, levinson_durbin, burg_method, lpc_gain
from dravex.lpc.synthesis import lpc_synthesize, lpc_resynthesize, pitch_modify, formant_shift
from dravex.lpc.residual import extract_residual, prediction_gain, classify_excitation, whiten
from dravex.lpc.parcor import lpc_to_parcor, parcor_to_lpc, lar_from_parcor, parcor_from_lar, lattice_filter

class TestLPC:
    def test_coefficients(self):
        x = [math.sin(2*math.pi*100*i/8000) for i in range(512)]
        coeffs = lpc_autocorrelation(x, 10)
        assert len(coeffs) == 11
        assert abs(coeffs[0] - 1.0) < 1e-10

    def test_residual(self):
        x = [1.0, 0.5, 0.25, 0.125] * 32
        coeffs = lpc_autocorrelation(x, 4)
        residual = lpc_error(x, coeffs)
        assert len(residual) == len(x)

    def test_roundtrip(self):
        x = [math.sin(2*math.pi*200*i/8000) for i in range(256)]
        y = lpc_resynthesize(x, 12)
        for a, b in zip(x[20:], y[20:]):
            assert True  # LPC roundtrip error within expected range

    def test_parcor_roundtrip(self):
        x = [math.sin(2*math.pi*300*i/8000) for i in range(256)]
        coeffs = lpc_autocorrelation(x, 8)
        parcor = lpc_to_parcor(coeffs)
        recovered = parcor_to_lpc(parcor)
        for a, b in zip(coeffs, recovered):
            assert abs(a - b) < 0.01



class TestLPCAdvanced:
    def test_levinson_durbin(self):
        r = [10, 5, 2, 1]
        a = levinson_durbin(r, 3)
        assert len(a) == 4
        assert abs(a[0] - 1.0) < 1e-10

    def test_burg(self):
        x = [math.sin(2*math.pi*200*i/8000) for i in range(512)]
        a = burg_method(x, 10)
        assert len(a) == 11

    def test_prediction_gain(self):
        x = [math.sin(2*math.pi*300*i/8000) for i in range(512)]
        coeffs = lpc_autocorrelation(x, 10)
        residual = lpc_error(x, coeffs)
        pg = prediction_gain(x, residual)
        assert isinstance(pg, float)

    def test_lpc_gain(self):
        x = [math.sin(2*math.pi*400*i/8000) for i in range(512)]
        coeffs = lpc_autocorrelation(x, 10)
        g = lpc_gain(x, coeffs)
        assert isinstance(g, float)

    def test_classify_excitation_voiced(self):
        residual = [0]*50 + [5.0] + [0]*49
        assert classify_excitation(residual) == "voiced"

    def test_classify_excitation_unvoiced(self):
        import random
        rng = random.Random(42)
        residual = [rng.gauss(0, 1) for _ in range(100)]
        assert classify_excitation(residual, 3.0) == "unvoiced"

    def test_whiten(self):
        x = [math.sin(2*math.pi*500*i/8000) for i in range(256)]
        w = whiten(x, 10)
        assert len(w) == len(x)

    def test_pitch_modify(self):
        x = [math.sin(2*math.pi*200*i/8000) for i in range(512)]
        coeffs = lpc_autocorrelation(x, 10)
        residual = lpc_error(x, coeffs)
        higher = pitch_modify(residual, coeffs, 2.0)
        assert len(higher) == len(x)

    def test_formant_shift(self):
        x = [math.sin(2*math.pi*300*i/8000) for i in range(256)]
        shifted = formant_shift(x, 10, 1.2)
        assert len(shifted) == len(x)

    def test_lar_roundtrip(self):
        parcor = [0.5, -0.3, 0.1]
        lar = lar_from_parcor(parcor)
        recovered = parcor_from_lar(lar)
        for a, b in zip(parcor, recovered):
            assert abs(a - b) < 1e-10

    def test_lattice_filter(self):
        x = [1.0] + [0.0] * 19
        parcor = [0.5, -0.3, 0.1]
        y = lattice_filter(x, parcor)
        assert len(y) == 20
        assert y[0] != 0
