"""A-law companding (European standard).

Implements ITU-T G.711 A-law encoding/decoding for 8-bit
PCM compression used in European telephone networks.
"""
import math

A_CONST = 87.6

def alaw_encode(x, A=A_CONST):
    """A-law compression."""
    result = []
    log_A = math.log(A)
    for s in x:
        sign = 1.0 if s >= 0 else -1.0
        mag = abs(s)
        if mag < 1.0 / A:
            compressed = A * mag / (1 + log_A)
        else:
            compressed = (1 + math.log(A * mag)) / (1 + log_A)
        result.append(sign * compressed)
    return result

def alaw_decode(y, A=A_CONST):
    """A-law expansion (inverse)."""
    result = []
    log_A = math.log(A)
    for s in y:
        sign = 1.0 if s >= 0 else -1.0
        mag = abs(s)
        if mag < 1.0 / (1 + log_A):
            expanded = mag * (1 + log_A) / A
        else:
            expanded = math.exp(mag * (1 + log_A) - 1) / A
        result.append(sign * expanded)
    return result

def alaw_quantize(x, bits=8, A=A_CONST):
    """A-law encode + quantize."""
    compressed = alaw_encode(x, A)
    levels = 2 ** bits
    return [max(0, min(levels - 1, int((s + 1) / 2 * (levels - 1)))) for s in compressed]

def alaw_dequantize(codes, bits=8, A=A_CONST):
    """Dequantize + A-law decode."""
    levels = 2 ** bits
    compressed = [(c / (levels - 1)) * 2 - 1 for c in codes]
    return alaw_decode(compressed, A)
