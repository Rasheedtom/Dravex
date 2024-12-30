"""Hann (Hanning) window function.

The Hann window provides good frequency resolution with
moderate spectral leakage. Main lobe width is 4 bins.
"""
import math
from dravex.constants import TAU

def hann(n, symmetric=True):
    """Generate Hann window of length n."""
    if n <= 1:
        return [1.0]
    M = n - 1 if symmetric else n
    return [0.5 * (1 - math.cos(TAU * i / M)) for i in range(n)]

def hann_periodic(n):
    """Periodic Hann window for STFT."""
    return hann(n, symmetric=False)

def apply_hann(x):
    """Apply Hann window to signal."""
    w = hann(len(x))
    return [s * wi for s, wi in zip(x, w)]

def hann_poisson(n, alpha=2.0):
    """Hann-Poisson hybrid window (better sidelobe decay)."""
    M = n - 1 if n > 1 else 1
    result = []
    for i in range(n):
        hann_part = 0.5 * (1 - math.cos(TAU * i / M))
        poisson_part = math.exp(-alpha * abs(M - 2 * i) / M)
        result.append(hann_part * poisson_part)
    return result

def cosine_sum_window(n, coefficients):
    """General cosine-sum window: w[n] = sum(a_k * cos(2*pi*k*n/N))."""
    M = n - 1 if n > 1 else 1
    result = []
    for i in range(n):
        s = 0.0
        for k, ak in enumerate(coefficients):
            sign = (-1) ** k
            s += sign * ak * math.cos(TAU * k * i / M)
        result.append(s)
    return result
