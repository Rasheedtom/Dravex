"""Tests for compression codecs."""
from dravex.compress.mulaw import mulaw_encode, mulaw_decode
from dravex.compress.alaw import alaw_encode, alaw_decode
from dravex.compress.adpcm import adpcm_encode, adpcm_decode
from dravex.compress.quantize import uniform_quantize, sqnr
from dravex.compress.dither import rectangular_dither
from dravex.compress.mulaw import mulaw_encode, mulaw_decode, mulaw_quantize, mulaw_dequantize, sndr_mulaw
from dravex.compress.alaw import alaw_encode, alaw_decode, alaw_quantize, alaw_dequantize
from dravex.compress.adpcm import adpcm_encode, adpcm_decode, dpcm_encode, dpcm_decode
from dravex.compress.quantize import uniform_quantize, log_quantize, quantization_noise, sqnr, midrise_quantize
from dravex.compress.dither import rectangular_dither, triangular_dither, noise_shaped_dither
import math

class TestCompress:
    def test_mulaw_roundtrip(self):
        x = [0.5, -0.3, 0.8]
        encoded = mulaw_encode(x)
        decoded = mulaw_decode(encoded)
        for a, b in zip(x, decoded):
            assert abs(a - b) < 0.01

    def test_alaw_roundtrip(self):
        x = [0.5, -0.3]
        encoded = alaw_encode(x)
        decoded = alaw_decode(encoded)
        for a, b in zip(x, decoded):
            assert abs(a - b) < 0.05

    def test_adpcm(self):
        x = [0.1 * i for i in range(20)]
        codes = adpcm_encode(x)
        decoded = adpcm_decode(codes)
        assert len(decoded) == len(x)

    def test_uniform_quantize(self):
        x = [0.5, -0.5, 0.0]
        q = uniform_quantize(x, 8)
        assert len(q) == 3

    def test_sqnr(self):
        assert abs(sqnr(16) - 98.08) < 0.1

    def test_dither(self):
        x = [0.5, 0.3, 0.1]
        d = rectangular_dither(x, 8)
        assert len(d) == 3



class TestCompressAdvanced:
    def test_mulaw_quantize_roundtrip(self):
        x = [0.5, -0.3, 0.8, -0.1]
        codes = mulaw_quantize(x, 8)
        decoded = mulaw_dequantize(codes, 8)
        for a, b in zip(x, decoded):
            assert abs(a - b) < 0.1

    def test_alaw_quantize(self):
        x = [0.1, -0.5, 0.9]
        codes = alaw_quantize(x, 8)
        assert all(0 <= c < 256 for c in codes)

    def test_dpcm_roundtrip(self):
        x = [float(i) / 10 for i in range(20)]
        diffs = dpcm_encode(x)
        decoded = dpcm_decode(diffs)
        for a, b in zip(x, decoded):
            assert abs(a - b) < 1e-10

    def test_log_quantize(self):
        x = [0.01, 0.1, 0.5, 0.9]
        q = log_quantize(x, 8)
        assert len(q) == 4

    def test_quantization_noise(self):
        x = [0.5, 0.3]
        q = uniform_quantize(x, 8)
        noise = quantization_noise(x, q)
        assert all(abs(n) < 0.01 for n in noise)

    def test_sqnr_16bit(self):
        assert abs(sqnr(16) - 98.08) < 0.1

    def test_sndr_mulaw(self):
        sndr = sndr_mulaw(8)
        assert sndr > 30

    def test_midrise(self):
        x = [0.0, 0.5, -0.5]
        q = midrise_quantize(x, 4)
        assert len(q) == 3

    def test_triangular_dither(self):
        x = [0.5] * 10
        d = triangular_dither(x, 8)
        assert len(d) == 10

    def test_noise_shaped_dither(self):
        x = [math.sin(0.1*i) for i in range(100)]
        d = noise_shaped_dither(x, 8)
        assert len(d) == 100
