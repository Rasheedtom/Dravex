"""Kaiser window function."""
import math
from dravex.constants import PI

def kaiser(n, beta=8.6):
    """Generate Kaiser window of length n with shape parameter beta."""
    if n <= 1:
        return [1.0]
    M = n - 1
    result = []
    for i in range(n):
        x = 2.0 * i / M - 1.0
        arg = beta * math.sqrt(max(0, 1 - x * x))
        result.append(_bessel_i0(arg) / _bessel_i0(beta))
    return result

def _bessel_i0(x):
    """Modified Bessel function I0 via series expansion."""
    result = 1.0
    term = 1.0
    for k in range(1, 30):
        term *= (x / (2 * k)) ** 2
        result += term
        if term < 1e-15:
            break
    return result

def kaiser_beta_from_atten(atten_db):
    """Estimate Kaiser beta from desired stopband attenuation."""
    if atten_db > 50:
        return 0.1102 * (atten_db - 8.7)
    elif atten_db >= 21:
        return 0.5842 * (atten_db - 21) ** 0.4 + 0.07886 * (atten_db - 21)
    return 0.0

def kaiser_order(transition_width, atten_db, sample_rate):
    """Estimate minimum filter order for Kaiser window design."""
    delta_f = transition_width / sample_rate
    if atten_db > 21:
        order = (atten_db - 7.95) / (2.285 * 2 * PI * delta_f)
    else:
        order = 5.79 / (2 * PI * delta_f)
    return max(1, math.ceil(order))

def apply_kaiser(x, beta=8.6):
    """Apply Kaiser window to signal."""
    w = kaiser(len(x), beta)
    return [s * wi for s, wi in zip(x, w)]
