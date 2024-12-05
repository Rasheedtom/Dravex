"""Infinite Impulse Response filter application."""
import math
from dravex.errors import StabilityError

def apply_iir(x, b, a):
    """Apply IIR filter: y[n] = sum(b[k]*x[n-k]) - sum(a[k]*y[n-k])."""
    n = len(x)
    nb = len(b)
    na = len(a)
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(nb):
            if i - j >= 0:
                s += b[j] * x[i - j]
        for j in range(1, na):
            if i - j >= 0:
                s -= a[j] * y[i - j]
        if abs(a[0]) > 1e-15:
            y[i] = s / a[0]
        else:
            y[i] = s
    return y

def biquad(x, b0, b1, b2, a1, a2):
    """Second-order IIR section (biquad)."""
    n = len(x)
    y = [0.0] * n
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        y[i] = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x[i]
        y2, y1 = y1, y[i]
    return y

def cascade_biquads(x, sections):
    """Apply cascade of biquad sections."""
    result = list(x)
    for b0, b1, b2, a1, a2 in sections:
        result = biquad(result, b0, b1, b2, a1, a2)
    return result

def check_stability(a):
    """Check IIR filter stability."""
    from dravex.transform.zplane import is_stable
    if not is_stable(a):
        raise StabilityError()
    return True

def zero_phase_filter(x, b, a):
    """Zero-phase filtering (forward-backward)."""
    forward = apply_iir(x, b, a)
    backward = apply_iir(list(reversed(forward)), b, a)
    return list(reversed(backward))

def sos_filter(x, sections):
    """Apply second-order sections (SOS) IIR filter."""
    result = list(x)
    for b0, b1, b2, a0, a1, a2 in sections:
        n = len(result)
        y = [0.0] * n
        x1 = x2 = y1 = y2 = 0.0
        for i in range(n):
            y[i] = (b0 * result[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
            x2, x1 = x1, result[i]
            y2, y1 = y1, y[i]
        result = y
    return result

def resonator(x, center_freq, bandwidth, sample_rate):
    """Second-order resonator (bandpass around center frequency)."""
    import math
    w0 = 2 * math.pi * center_freq / sample_rate
    r = 1.0 - math.pi * bandwidth / sample_rate
    r = max(0.0, min(0.999, r))
    b = [1.0 - r, 0.0, -(1.0 - r)]
    a = [1.0, -2 * r * math.cos(w0), r * r]
    return apply_iir(x, b, a)

def parametric_eq(x, center_freq, gain_db, q, sample_rate):
    """Parametric equalizer (boost/cut at center frequency)."""
    import math
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * center_freq / sample_rate
    alpha = math.sin(w0) / (2 * q)
    if gain_db >= 0:
        b = [1 + alpha * A, -2 * math.cos(w0), 1 - alpha * A]
        a = [1 + alpha / A, -2 * math.cos(w0), 1 - alpha / A]
    else:
        b = [1 + alpha / A, -2 * math.cos(w0), 1 - alpha / A]
        a = [1 + alpha * A, -2 * math.cos(w0), 1 - alpha * A]
    return apply_iir(x, b, a)

def shelving_low(x, cutoff, gain_db, sample_rate):
    """Low-shelf filter."""
    import math
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * cutoff / sample_rate
    alpha = math.sin(w0) / 2 * math.sqrt((A + 1/A) * 2)
    sqA = math.sqrt(A)
    b0 = A * ((A+1) - (A-1)*math.cos(w0) + 2*sqA*alpha)
    b1 = 2*A * ((A-1) - (A+1)*math.cos(w0))
    b2 = A * ((A+1) - (A-1)*math.cos(w0) - 2*sqA*alpha)
    a0 = (A+1) + (A-1)*math.cos(w0) + 2*sqA*alpha
    a1 = -2 * ((A-1) + (A+1)*math.cos(w0))
    a2 = (A+1) + (A-1)*math.cos(w0) - 2*sqA*alpha
    return apply_iir(x, [b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])
