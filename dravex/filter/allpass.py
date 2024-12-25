"""Allpass filter for phase manipulation."""
import math
from dravex.constants import PI

def allpass_first_order(x, a):
    """First-order allpass: y[n] = a*x[n] + x[n-1] - a*y[n-1]."""
    n = len(x)
    y = [0.0] * n
    y[0] = a * x[0]
    for i in range(1, n):
        y[i] = a * x[i] + x[i-1] - a * y[i-1]
    return y

def allpass_second_order(x, r, theta):
    """Second-order allpass with pole at r*e^(j*theta)."""
    a1 = -2 * r * math.cos(theta)
    a2 = r * r
    n = len(x)
    y = [0.0] * n
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        y[i] = a2 * x[i] + a1 * x1 + x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x[i]
        y2, y1 = y1, y[i]
    return y

def phase_shift(x, shift_radians, sample_rate):
    """Apply constant phase shift using allpass cascade."""
    target_freq = sample_rate / 8
    w0 = 2 * PI * target_freq / sample_rate
    r = 0.9
    theta = w0 + shift_radians / 2
    return allpass_second_order(x, r, theta)

def allpass_reverb(x, delays, gains):
    """Simple allpass reverberator."""
    result = list(x)
    for delay, gain in zip(delays, gains):
        n = len(result)
        output = [0.0] * n
        for i in range(n):
            if i >= delay:
                output[i] = -gain * result[i] + result[i - delay] + gain * output[i - delay]
            else:
                output[i] = -gain * result[i]
        result = output
    return result

def dispersion_filter(x, order=4, coefficient=0.5):
    """Cascaded allpass for dispersion simulation."""
    result = list(x)
    for _ in range(order):
        result = allpass_first_order(result, coefficient)
    return result
