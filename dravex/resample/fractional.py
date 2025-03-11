"""Fractional delay and sample-accurate time shifting."""
import math
from dravex.constants import sinc as sinc_fn

def fractional_delay(x, delay_samples, window_size=8):
    """Apply fractional sample delay using sinc interpolation."""
    n = len(x)
    result = [0.0] * n
    int_delay = int(math.floor(delay_samples))
    frac = delay_samples - int_delay
    for i in range(n):
        val = 0.0
        for k in range(-window_size, window_size + 1):
            idx = i - int_delay - k
            if 0 <= idx < n:
                val += x[idx] * sinc_fn(frac + k) * _hann_window_val(frac + k, window_size)
        result[i] = val
    return result

def _hann_window_val(x, half_width):
    if abs(x) >= half_width:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * x / half_width))

def variable_delay(x, delay_curve, window_size=4):
    """Apply time-varying delay (chorus/flanger effect)."""
    n = len(x)
    result = [0.0] * n
    for i in range(n):
        d = delay_curve[i] if i < len(delay_curve) else 0
        int_d = int(math.floor(d))
        frac = d - int_d
        idx = i - int_d
        if 0 <= idx < n:
            if idx + 1 < n:
                result[i] = x[idx] * (1 - frac) + x[idx + 1] * frac
            else:
                result[i] = x[idx]
    return result

def pitch_shift_td(x, semitones, sample_rate=44100, frame_size=1024):
    """Simple time-domain pitch shifting via resampling + overlap."""
    ratio = 2.0 ** (semitones / 12.0)
    from dravex.resample.interpolate import resample_ratio
    stretched = resample_ratio(x, 1.0 / ratio)
    if len(stretched) > len(x):
        return stretched[:len(x)]
    return stretched + [0.0] * (len(x) - len(stretched))
