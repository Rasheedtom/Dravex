"""μ-law companding for telecommunications.

Implements ITU-T G.711 μ-law encoding/decoding for 8-bit
PCM compression used in North American telephone networks.
"""
import math
from dravex.constants import DEFAULT_EPSILON

MU = 255

def mulaw_encode(x, mu=MU):
    """μ-law compression: F(x) = sgn(x) * ln(1 + μ|x|) / ln(1 + μ)."""
    result = []
    for s in x:
        sign = 1.0 if s >= 0 else -1.0
        compressed = sign * math.log(1 + mu * abs(s)) / math.log(1 + mu)
        result.append(compressed)
    return result

def mulaw_decode(y, mu=MU):
    """μ-law expansion (inverse)."""
    result = []
    for s in y:
        sign = 1.0 if s >= 0 else -1.0
        expanded = sign * ((1 + mu) ** abs(s) - 1) / mu
        result.append(expanded)
    return result

def mulaw_quantize(x, bits=8, mu=MU):
    """μ-law encode + quantize to n bits."""
    compressed = mulaw_encode(x, mu)
    levels = 2 ** bits
    quantized = []
    for s in compressed:
        q = int((s + 1) / 2 * (levels - 1))
        q = max(0, min(levels - 1, q))
        quantized.append(q)
    return quantized

def mulaw_dequantize(codes, bits=8, mu=MU):
    """Dequantize + μ-law decode."""
    levels = 2 ** bits
    compressed = [(c / (levels - 1)) * 2 - 1 for c in codes]
    return mulaw_decode(compressed, mu)

def sndr_mulaw(bits=8, mu=MU):
    """Signal-to-noise-and-distortion ratio for μ-law at given bits."""
    return 6.02 * bits + 4.77 - 10 * math.log10(3) + 20 * math.log10(math.log(1 + mu))
