"""Complete sample rate conversion pipeline."""
import math

def convert_sample_rate(x, from_rate, to_rate, quality="medium"):
    """High-level sample rate conversion."""
    if from_rate == to_rate:
        return list(x)
    ratio = to_rate / from_rate
    if quality == "low":
        from dravex.resample.interpolate import resample_ratio
        return resample_ratio(x, ratio)
    elif quality == "medium":
        from dravex.resample.sinc_resample import bandwidth_limited_resample
        return bandwidth_limited_resample(x, from_rate, to_rate, window_size=8)
    else:
        from dravex.resample.sinc_resample import bandwidth_limited_resample
        return bandwidth_limited_resample(x, from_rate, to_rate, window_size=32)

def cd_to_telephone(x):
    """Convert 44100 Hz to 8000 Hz."""
    return convert_sample_rate(x, 44100, 8000, "medium")

def telephone_to_cd(x):
    """Convert 8000 Hz to 44100 Hz."""
    return convert_sample_rate(x, 8000, 44100, "medium")

def standard_rates():
    """List of standard audio sample rates."""
    return [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000]

def nearest_standard_rate(rate):
    """Find nearest standard sample rate."""
    rates = standard_rates()
    return min(rates, key=lambda r: abs(r - rate))

def gcd(a, b):
    """Greatest common divisor for rational resampling."""
    while b:
        a, b = b, a % b
    return a

def rational_resample_factors(from_rate, to_rate):
    """Compute up/down factors for rational resampling."""
    g = gcd(from_rate, to_rate)
    return to_rate // g, from_rate // g
