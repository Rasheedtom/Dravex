"""Butterworth filter design.

Butterworth filters have maximally flat magnitude response
in the passband, making them ideal for general-purpose filtering.
"""
import math
from dravex.constants import PI, TAU

def butterworth_lowpass(order, cutoff, sample_rate):
    """Design Butterworth lowpass filter coefficients."""
    wc = math.tan(PI * cutoff / sample_rate)
    b_coeffs = [0.0] * (order + 1)
    a_coeffs = [0.0] * (order + 1)
    if order == 1:
        b_coeffs[0] = wc / (1 + wc)
        b_coeffs[1] = wc / (1 + wc)
        a_coeffs[0] = 1.0
        a_coeffs[1] = (wc - 1) / (1 + wc)
    elif order == 2:
        k = wc * wc
        d = 1 + math.sqrt(2) * wc + k
        b_coeffs[0] = k / d
        b_coeffs[1] = 2 * k / d
        b_coeffs[2] = k / d
        a_coeffs[0] = 1.0
        a_coeffs[1] = 2 * (k - 1) / d
        a_coeffs[2] = (1 - math.sqrt(2) * wc + k) / d
    else:
        sections = butterworth_sections(order, cutoff, sample_rate)
        return sections
    return b_coeffs, a_coeffs

def butterworth_highpass(order, cutoff, sample_rate):
    """Design Butterworth highpass filter."""
    wc = math.tan(PI * cutoff / sample_rate)
    if order == 2:
        d = 1 + math.sqrt(2) * wc + wc * wc
        b = [1/d, -2/d, 1/d]
        a = [1.0, 2*(wc*wc - 1)/d, (1 - math.sqrt(2)*wc + wc*wc)/d]
        return b, a
    return [1.0], [1.0]

def butterworth_sections(order, cutoff, sample_rate):
    """Return cascade of biquad sections for higher-order Butterworth."""
    wc = math.tan(PI * cutoff / sample_rate)
    sections = []
    for k in range(order // 2):
        theta = PI * (2 * k + 1) / (2 * order)
        d = 1 + 2 * wc * math.cos(theta) + wc * wc
        b0 = wc * wc / d
        b1 = 2 * wc * wc / d
        b2 = wc * wc / d
        a1 = 2 * (wc * wc - 1) / d
        a2 = (1 - 2 * wc * math.cos(theta) + wc * wc) / d
        sections.append((b0, b1, b2, a1, a2))
    if order % 2 == 1:
        d = 1 + wc
        sections.append((wc / d, wc / d, 0.0, (wc - 1) / d, 0.0))
    return sections

def butterworth_order(passband_freq, stopband_freq, passband_ripple_db, stopband_atten_db, sample_rate):
    """Estimate minimum Butterworth filter order."""
    wp = math.tan(PI * passband_freq / sample_rate)
    ws = math.tan(PI * stopband_freq / sample_rate)
    num = math.log10((10**(stopband_atten_db/10) - 1) / (10**(passband_ripple_db/10) - 1))
    den = 2 * math.log10(ws / wp)
    return max(1, math.ceil(num / den))

def butterworth_bandpass(order, low_cut, high_cut, sample_rate):
    """Butterworth bandpass via LP to BP transform."""
    import math
    wl = math.tan(PI * low_cut / sample_rate)
    wh = math.tan(PI * high_cut / sample_rate)
    bw = wh - wl
    w0 = math.sqrt(wl * wh)
    if order == 2:
        Q = w0 / bw
        d = 1 + w0 / Q + w0 * w0
        b0 = (bw * w0) / d
        b = [b0, 0, -b0]
        a = [1.0, 2*(w0*w0 - 1)/d, (1 - w0/Q + w0*w0)/d]
        return b, a
    return [1.0], [1.0]

def butterworth_response(b, a, n_points=256, sample_rate=44100):
    """Compute magnitude response of Butterworth filter."""
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
        den_mag = den_r*den_r + den_i*den_i
        if den_mag < 1e-15:
            mags.append(1e6)
        else:
            h_r = (num_r*den_r + num_i*den_i) / den_mag
            h_i = (num_i*den_r - num_r*den_i) / den_mag
            mags.append(math.sqrt(h_r*h_r + h_i*h_i))
    return freqs, mags
