"""Dithering for quantization noise shaping."""
import math
import random as _random

def rectangular_dither(x, bits, rng=None):
    """Add rectangular (uniform) dither before quantization."""
    if rng is None:
        rng = _random.Random(42)
    levels = 2 ** bits
    delta = 2.0 / levels
    dithered = [s + (rng.random() - 0.5) * delta for s in x]
    from dravex.compress.quantize import uniform_quantize
    return uniform_quantize(dithered, bits)

def triangular_dither(x, bits, rng=None):
    """Add triangular (TPDF) dither before quantization."""
    if rng is None:
        rng = _random.Random(42)
    levels = 2 ** bits
    delta = 2.0 / levels
    dithered = []
    for s in x:
        d = (rng.random() + rng.random() - 1.0) * delta
        dithered.append(s + d)
    from dravex.compress.quantize import uniform_quantize
    return uniform_quantize(dithered, bits)

def noise_shaped_dither(x, bits, rng=None):
    """First-order noise-shaped dithering."""
    if rng is None:
        rng = _random.Random(42)
    levels = 2 ** bits
    half = levels // 2
    error = 0.0
    result = []
    for s in x:
        dither = (rng.random() - 0.5) / half
        val = s - error + dither
        q = int(round(val * half))
        q = max(-half, min(half - 1, q))
        quantized = q / half
        error = quantized - s
        result.append(quantized)
    return result

def dither_amount(bits):
    """Recommended dither amplitude (peak-to-peak) for given bit depth."""
    return 2.0 / (2 ** bits)
