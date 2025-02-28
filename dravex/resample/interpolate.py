"""Signal interpolation methods."""
import math
from dravex.constants import sinc, DEFAULT_EPSILON

def linear_interpolate(x, factor):
    """Upsample by integer factor using linear interpolation."""
    n = len(x)
    if n < 2 or factor < 1:
        return list(x)
    out = []
    for i in range(n - 1):
        out.append(x[i])
        for k in range(1, factor):
            t = k / factor
            out.append(x[i] * (1 - t) + x[i + 1] * t)
    out.append(x[-1])
    return out

def cubic_interpolate(x, factor):
    """Upsample using cubic (Hermite) interpolation.

    Requires at least 4 input samples for cubic evaluation.
    """
    """Upsample using cubic (Hermite) interpolation."""
    n = len(x)
    if n < 4 or factor < 1:
        return list(x)
    out = []
    for i in range(1, n - 2):
        out.append(x[i])
        for k in range(1, factor):
            t = k / factor
            a = x[i - 1]
            b = x[i]
            c = x[i + 1]
            d = x[i + 2]
            val = _cubic_hermite(a, b, c, d, t)
            out.append(val)
    out.append(x[-2])
    return out

def _cubic_hermite(y0, y1, y2, y3, t):
    a0 = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3
    a1 = y0 - 2.5*y1 + 2*y2 - 0.5*y3
    a2 = -0.5*y0 + 0.5*y2
    a3 = y1
    return a0*t*t*t + a1*t*t + a2*t + a3

def resample_ratio(x, ratio):
    """Resample signal by arbitrary ratio using linear interpolation."""
    n = len(x)
    new_n = max(1, int(n * ratio))
    result = []
    for i in range(new_n):
        pos = i / ratio
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < n:
            result.append(x[idx] * (1 - frac) + x[idx + 1] * frac)
        elif idx < n:
            result.append(x[idx])
        else:
            result.append(0.0)
    return result

def zero_order_hold(x, factor):
    """Upsample by repeating each sample."""
    out = []
    for s in x:
        out.extend([s] * factor)
    return out
