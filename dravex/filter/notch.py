"""Notch (band-reject) filter for removing specific frequencies."""
import math
from dravex.constants import PI

def notch_filter(center_freq, bandwidth, sample_rate):
    """Design second-order IIR notch filter.

    Quality factor Q = center_freq / bandwidth determines
    the sharpness of the frequency rejection.
    """
    """Design second-order notch filter."""
    w0 = 2 * PI * center_freq / sample_rate
    bw = 2 * PI * bandwidth / sample_rate
    r = 1.0 - bw / 2
    r = max(0.0, min(0.999, r))
    b = [1.0, -2 * math.cos(w0), 1.0]
    a = [1.0, -2 * r * math.cos(w0), r * r]
    return b, a

def apply_notch(x, center_freq, bandwidth, sample_rate):
    """Apply notch filter to signal."""
    b, a = notch_filter(center_freq, bandwidth, sample_rate)
    from dravex.filter.iir import apply_iir
    return apply_iir(x, b, a)

def comb_notch(x, fundamental_freq, n_harmonics, bandwidth, sample_rate):
    """Apply notch filters at fundamental and harmonics."""
    result = list(x)
    from dravex.filter.iir import apply_iir
    for h in range(1, n_harmonics + 1):
        freq = fundamental_freq * h
        if freq >= sample_rate / 2:
            break
        b, a = notch_filter(freq, bandwidth, sample_rate)
        result = apply_iir(result, b, a)
    return result

def adaptive_notch(x, step_size=0.01, sample_rate=44100):
    """LMS-based adaptive notch filter."""
    n = len(x)
    y = [0.0] * n
    w = [0.0, 0.0]
    for i in range(2, n):
        ref = [x[i-1], x[i-2]]
        est = w[0] * ref[0] + w[1] * ref[1]
        y[i] = x[i] - est
        w[0] += step_size * y[i] * ref[0]
        w[1] += step_size * y[i] * ref[1]
    return y

def twin_t_notch(center_freq, sample_rate):
    """Twin-T notch filter design."""
    import math
    w0 = 2 * math.pi * center_freq / sample_rate
    cos_w = math.cos(w0)
    b = [1.0, -2*cos_w, 1.0]
    r = 0.95
    a = [1.0, -2*r*cos_w, r*r]
    return b, a

def multi_notch(x, frequencies, bandwidth, sample_rate):
    """Apply multiple notch filters at specified frequencies."""
    result = list(x)
    from dravex.filter.iir import apply_iir
    for freq in frequencies:
        b, a = notch_filter(freq, bandwidth, sample_rate)
        result = apply_iir(result, b, a)
    return result

def notch_quality_factor(center_freq, bandwidth):
    """Compute quality factor Q from center frequency and bandwidth."""
    if bandwidth < 1e-12:
        return float('inf')
    return center_freq / bandwidth

def peak_filter(center_freq, gain_db, bandwidth, sample_rate):
    """Peak (bell) filter - inverse of notch for boosting."""
    import math
    w0 = 2 * math.pi * center_freq / sample_rate
    A = 10 ** (gain_db / 40.0)
    bw = 2 * math.pi * bandwidth / sample_rate
    alpha = math.sin(w0) * math.sinh(math.log(2)/2 * bw * w0 / math.sin(w0)) if math.sin(w0) > 1e-12 else 0.5
    b = [1 + alpha * A, -2 * math.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * math.cos(w0), 1 - alpha / A]
    return b, a
