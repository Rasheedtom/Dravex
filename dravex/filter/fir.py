"""Finite Impulse Response filter design and application."""
import math
from dravex.constants import PI, TAU, sinc

def fir_lowpass(order, cutoff, sample_rate):
    """Design FIR lowpass filter using windowed-sinc method."""
    fc = cutoff / sample_rate
    M = order
    h = []
    for n in range(M + 1):
        if n == M // 2:
            h.append(2 * fc)
        else:
            h.append(sinc(2 * fc * (n - M / 2)) * 2 * fc)
    w = [0.54 - 0.46 * math.cos(TAU * n / M) for n in range(M + 1)]
    result = [hi * wi for hi, wi in zip(h, w)]
    dc_gain = sum(result)
    if abs(dc_gain) > 1e-12:
        result = [c / dc_gain for c in result]
    return result

def fir_highpass(order, cutoff, sample_rate):
    """Design FIR highpass via spectral inversion."""
    lp = fir_lowpass(order, cutoff, sample_rate)
    hp = [-c for c in lp]
    hp[len(hp) // 2] += 1.0
    return hp

def fir_bandpass(order, low_cut, high_cut, sample_rate):
    """Design FIR bandpass filter."""
    lp_high = fir_lowpass(order, high_cut, sample_rate)
    lp_low = fir_lowpass(order, low_cut, sample_rate)
    return [h - l for h, l in zip(lp_high, lp_low)]

def fir_bandstop(order, low_cut, high_cut, sample_rate):
    """Design FIR bandstop (notch) filter."""
    bp = fir_bandpass(order, low_cut, high_cut, sample_rate)
    bs = [-c for c in bp]
    bs[len(bs) // 2] += 1.0
    return bs

def apply_fir(signal_samples, coefficients):
    """Apply FIR filter via direct convolution."""
    n = len(signal_samples)
    m = len(coefficients)
    output = []
    for i in range(n):
        s = 0.0
        for j in range(m):
            idx = i - j
            if 0 <= idx < n:
                s += coefficients[j] * signal_samples[idx]
        output.append(s)
    return output

def fir_frequency_response(coefficients, n_points=512):
    """Compute frequency response of FIR filter."""
    from dravex.transform.zplane import frequency_response
    return frequency_response(coefficients, [1.0], n_points)

def fir_moving_average(window_size):
    """Moving average FIR coefficients."""
    return [1.0 / window_size] * window_size

def fir_differentiator(order):
    """FIR differentiator filter."""
    import math
    M = order
    h = []
    for n in range(M + 1):
        if n == M // 2:
            h.append(0.0)
        else:
            k = n - M / 2
            h.append(math.cos(math.pi * k) / k)
    return h

def fir_hilbert(order):
    """FIR Hilbert transformer coefficients."""
    import math
    M = order
    h = []
    for n in range(M + 1):
        k = n - M / 2
        if abs(k) < 1e-10:
            h.append(0.0)
        elif int(k) % 2 == 0:
            h.append(0.0)
        else:
            h.append(2.0 / (math.pi * k))
    return h

def linear_phase_check(h):
    """Check if FIR filter has linear phase (symmetric coefficients)."""
    n = len(h)
    for i in range(n // 2):
        if abs(h[i] - h[n - 1 - i]) > 1e-10:
            return False
    return True

def group_delay_fir(h):
    """Group delay of linear-phase FIR filter."""
    if linear_phase_check(h):
        return (len(h) - 1) / 2.0
    return -1.0
