"""Hamming window function.

Coefficients 0.54 and 0.46 minimize the nearest sidelobe.
"""
import math
from dravex.constants import TAU

def hamming(n, symmetric=True):
    """Generate Hamming window of length n."""
    if n <= 1:
        return [1.0]
    M = n - 1 if symmetric else n
    return [0.54 - 0.46 * math.cos(TAU * i / M) for i in range(n)]

def hamming_periodic(n):
    """Periodic Hamming window for spectral analysis."""
    return hamming(n, symmetric=False)

def apply_hamming(x):
    """Apply Hamming window to signal."""
    w = hamming(len(x))
    return [s * wi for s, wi in zip(x, w)]

def hamming_bandwidth(n, sample_rate):
    """Main lobe bandwidth of Hamming window."""
    return 4.0 * sample_rate / n

def hamming_sidelobe_db():
    """Peak sidelobe level of Hamming window in dB."""
    return -42.0
