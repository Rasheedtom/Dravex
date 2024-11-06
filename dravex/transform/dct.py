"""Discrete Cosine Transform (Type II and III).

DCT-II is the standard DCT used in JPEG and MP3 compression.
DCT-III is the exact inverse of DCT-II.
"""
import math
from dravex.constants import PI

def dct_ii(x):
    """DCT Type II (the standard DCT). O(N^2)."""
    N = len(x)
    result = []
    for k in range(N):
        s = 0.0
        for n in range(N):
            s += x[n] * math.cos(PI * k * (2 * n + 1) / (2 * N))
        result.append(s * 2)
    return result

def dct_iii(X):
    """DCT Type III (inverse of DCT-II)."""
    N = len(X)
    result = []
    for n in range(N):
        s = X[0] / 2
        for k in range(1, N):
            s += X[k] * math.cos(PI * k * (2 * n + 1) / (2 * N))
        result.append(s / N)
    return result

def dct_fast(x):
    """Fast DCT via FFT for power-of-2 lengths."""
    N = len(x)
    reordered = [0.0] * N
    for i in range(N // 2):
        reordered[i] = x[2 * i]
        reordered[N - 1 - i] = x[2 * i + 1]
    from dravex.transform.fft import fft
    fft_result = fft(reordered)
    result = []
    for k in range(N):
        re, im = fft_result[k]
        angle = -PI * k / (2 * N)
        val = re * math.cos(angle) - im * math.sin(angle)
        result.append(val * 2)
    return result

def idct(X):
    """Inverse DCT (same as DCT-III with scaling)."""
    return dct_iii(X)

def dct_energy_compaction(x, keep_ratio=0.1):
    """Keep only top fraction of DCT coefficients."""
    coeffs = dct_ii(x)
    n_keep = max(1, int(len(coeffs) * keep_ratio))
    compressed = coeffs[:n_keep] + [0.0] * (len(coeffs) - n_keep)
    return dct_iii(compressed)

def dct_iv(x):
    """DCT Type IV (used in MDCT)."""
    import math
    N = len(x)
    result = []
    for k in range(N):
        s = 0.0
        for n in range(N):
            s += x[n] * math.cos(math.pi * (n + 0.5) * (k + 0.5) / N)
        result.append(s * math.sqrt(2.0 / N))
    return result

def mdct(x):
    """Modified DCT (MDCT) for audio coding."""
    import math
    N = len(x)
    M = N // 2
    result = []
    for k in range(M):
        s = 0.0
        for n in range(N):
            s += x[n] * math.cos(math.pi / M * (n + 0.5 + M / 2) * (k + 0.5))
        result.append(s)
    return result

def imdct(X, window_length):
    """Inverse MDCT."""
    import math
    M = len(X)
    N = window_length
    result = [0.0] * N
    for n in range(N):
        s = 0.0
        for k in range(M):
            s += X[k] * math.cos(math.pi / M * (n + 0.5 + M / 2) * (k + 0.5))
        result[n] = s / M
    return result
