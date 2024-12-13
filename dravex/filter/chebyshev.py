"""Chebyshev Type I and Type II filter design."""
import math
from dravex.constants import PI

def chebyshev1_lowpass(order, cutoff, ripple_db, sample_rate):
    """Design Chebyshev Type I lowpass filter."""
    eps = math.sqrt(10 ** (ripple_db / 10) - 1)
    wc = math.tan(PI * cutoff / sample_rate)
    if order == 2:
        gamma = math.asinh(1 / eps) / order
        sigma = -math.sinh(gamma)
        omega = math.cosh(gamma)
        p_re = sigma * math.sin(PI / 4)
        p_im = omega * math.cos(PI / 4)
        a0 = p_re * p_re + p_im * p_im
        b0_analog = a0
        a1_analog = -2 * p_re
        Omega = wc
        d = 1 + a1_analog * Omega + a0 * Omega * Omega
        b = [b0_analog * Omega * Omega / d,
             2 * b0_analog * Omega * Omega / d,
             b0_analog * Omega * Omega / d]
        a = [1.0,
             2 * (a0 * Omega * Omega - 1) / d,
             (1 - a1_analog * Omega + a0 * Omega * Omega) / d]
        if ripple_db > 3.0:
            scale = 10 ** (-ripple_db / 20)
            b = [bi * scale for bi in b]
        return b, a
    return [1.0], [1.0]

def chebyshev2_lowpass(order, cutoff, stopband_atten_db, sample_rate):
    """Design Chebyshev Type II lowpass (inverse Chebyshev)."""
    eps = 1.0 / math.sqrt(10 ** (stopband_atten_db / 10) - 1)
    wc = math.tan(PI * cutoff / sample_rate)
    if order == 2:
        k = wc * wc
        d = 1 + math.sqrt(2) * wc * eps + k * eps * eps
        b = [k / d, 2 * k / d, k / d]
        a = [1.0, 2 * (k * eps * eps - 1) / d, (1 - math.sqrt(2) * wc * eps + k * eps * eps) / d]
        return b, a
    return [1.0], [1.0]

def chebyshev_order(passband_freq, stopband_freq, ripple_db, atten_db, sample_rate):
    """Estimate minimum Chebyshev Type I order."""
    eps = math.sqrt(10 ** (ripple_db / 10) - 1)
    A = 10 ** (atten_db / 10)
    wp = math.tan(PI * passband_freq / sample_rate)
    ws = math.tan(PI * stopband_freq / sample_rate)
    ratio = ws / wp if wp > 0 else 1.0
    num = math.acosh(math.sqrt((A - 1) / (eps * eps)))
    den = math.acosh(ratio) if ratio > 1 else 1.0
    return max(1, math.ceil(num / den))

def chebyshev_response(b, a, n_points=256, sample_rate=44100):
    """Compute frequency response of Chebyshev filter."""
    import math
    freqs = []
    mags = []
    for i in range(n_points):
        f = i * sample_rate / (2 * n_points)
        w = 2 * PI * f / sample_rate
        freqs.append(f)
        num_r = sum(b[k] * math.cos(-w*k) for k in range(len(b)))
        num_i = sum(b[k] * math.sin(-w*k) for k in range(len(b)))
        den_r = sum(a[k] * math.cos(-w*k) for k in range(len(a)))
        den_i = sum(a[k] * math.sin(-w*k) for k in range(len(a)))
        den_mag = max(den_r*den_r + den_i*den_i, 1e-15)
        h_r = (num_r*den_r + num_i*den_i) / den_mag
        h_i = (num_i*den_r - num_r*den_i) / den_mag
        mags.append(math.sqrt(h_r*h_r + h_i*h_i))
    return freqs, mags

def ripple_to_epsilon(ripple_db):
    """Convert passband ripple in dB to epsilon."""
    import math
    return math.sqrt(10 ** (ripple_db / 10) - 1)
