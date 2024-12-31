"""Blackman window function."""
import math
from dravex.constants import TAU

def blackman(n, symmetric=True):
    """Generate Blackman window of length n."""
    if n <= 1:
        return [1.0]
    M = n - 1 if symmetric else n
    return [0.42 - 0.5 * math.cos(TAU * i / M) + 0.08 * math.cos(2 * TAU * i / M) for i in range(n)]

def blackman_harris(n, symmetric=True):
    """4-term Blackman-Harris window (minimum sidelobe)."""
    if n <= 1:
        return [1.0]
    M = n - 1 if symmetric else n
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    result = []
    for i in range(n):
        val = a0 - a1*math.cos(TAU*i/M) + a2*math.cos(2*TAU*i/M) - a3*math.cos(3*TAU*i/M)
        result.append(val)
    return result

def blackman_nuttall(n, symmetric=True):
    """Blackman-Nuttall window (very low sidelobes)."""
    if n <= 1:
        return [1.0]
    M = n - 1 if symmetric else n
    a0, a1, a2, a3 = 0.3635819, 0.4891775, 0.1365995, 0.0106411
    return [a0 - a1*math.cos(TAU*i/M) + a2*math.cos(2*TAU*i/M) - a3*math.cos(3*TAU*i/M) for i in range(n)]

def apply_blackman(x):
    """Apply Blackman window to signal."""
    w = blackman(len(x))
    return [s * wi for s, wi in zip(x, w)]

def flat_top(n, symmetric=True):
    """Flat-top window for amplitude calibration."""
    if n <= 1:
        return [1.0]
    M = n - 1 if symmetric else n
    a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    return [sum((-1)**k * a[k] * math.cos(k*TAU*i/M) for k in range(5)) for i in range(n)]

def exact_blackman(n, symmetric=True):
    """Exact Blackman window (slightly different coefficients)."""
    import math
    if n <= 1:
        return [1.0]
    M = n - 1 if symmetric else n
    a0 = 7938.0 / 18608.0
    a1 = 9240.0 / 18608.0
    a2 = 1430.0 / 18608.0
    return [a0 - a1*math.cos(2*math.pi*i/M) + a2*math.cos(4*math.pi*i/M) for i in range(n)]

def minimum_4term_blackman_harris(n, symmetric=True):
    """Minimum 4-term Blackman-Harris window."""
    import math
    if n <= 1:
        return [1.0]
    M = n - 1 if symmetric else n
    a = [0.35875, 0.48829, 0.14128, 0.01168]
    result = []
    for i in range(n):
        val = sum((-1)**k * a[k] * math.cos(2*math.pi*k*i/M) for k in range(4))
        result.append(val)
    return result

def window_mainlobe_width(window_func, n=1024, sample_rate=44100):
    """Estimate main lobe width in Hz."""
    import math
    w = window_func(n)
    from dravex.transform.fft import rfft
    mags = rfft(w + [0.0] * (4*n - n))
    peak = max(mags)
    threshold = peak * 0.5
    width = 0
    for i in range(len(mags)):
        if mags[i] >= threshold:
            width = i
    return width * sample_rate / (4 * n)
