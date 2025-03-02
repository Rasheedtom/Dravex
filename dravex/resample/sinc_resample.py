"""Sinc-based resampling for high-quality sample rate conversion."""
import math
from dravex.constants import sinc as sinc_fn, PI

def sinc_resample(x, new_length, window_size=16):
    """Windowed-sinc interpolation for arbitrary resampling."""
    n = len(x)
    if n == 0 or new_length == 0:
        return []
    ratio = n / new_length
    result = []
    for i in range(new_length):
        center = i * ratio
        val = 0.0
        for j in range(max(0, int(center) - window_size), min(n, int(center) + window_size + 1)):
            d = center - j
            w = sinc_fn(d) * _lanczos_window(d, window_size)
            val += x[j] * w
        result.append(val)
    return result

def _lanczos_window(x, a):
    """Lanczos window for sinc interpolation."""
    if abs(x) < 1e-12:
        return 1.0
    if abs(x) >= a:
        return 0.0
    return sinc_fn(x / a)

def bandwidth_limited_resample(x, old_rate, new_rate, window_size=16):
    """Resample between two sample rates with anti-aliasing."""
    new_length = int(len(x) * new_rate / old_rate)
    if new_rate < old_rate:
        from dravex.filter.fir import fir_lowpass, apply_fir
        cutoff = new_rate / 2 * 0.95
        h = fir_lowpass(64, cutoff, old_rate)
        x = apply_fir(x, h)
    return sinc_resample(x, new_length, window_size)

def sinc_upsample_2x(x, window_size=8):
    """High-quality 2x upsampling via sinc interpolation."""
    n = len(x)
    result = [0.0] * (2 * n)
    for i in range(n):
        result[2 * i] = x[i]
    for i in range(n - 1):
        val = 0.0
        for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
            d = (i + 0.5) - j
            val += x[j] * sinc_fn(d) * _lanczos_window(d, window_size)
        result[2 * i + 1] = val
    return result
