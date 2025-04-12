"""Signal mixing and frequency shifting."""
import math
from dravex.constants import TAU

def mix(signals, gains=None):
    """Mix multiple signals with optional gains."""
    if not signals:
        return []
    n = max(len(s) for s in signals)
    if gains is None:
        gains = [1.0 / len(signals)] * len(signals)
    result = [0.0] * n
    for sig, gain in zip(signals, gains):
        for i in range(min(len(sig), n)):
            result[i] += sig[i] * gain
    return result

def frequency_shift(x, shift_hz, sample_rate):
    """Shift spectrum by shift_hz using complex multiplication."""
    n = len(x)
    from dravex.transform.hilbert import analytic_signal
    analytic = analytic_signal(x)
    result = []
    for i in range(n):
        t = i / sample_rate
        cos_t = math.cos(TAU * shift_hz * t)
        sin_t = math.sin(TAU * shift_hz * t)
        re, im = analytic[i]
        shifted_re = re * cos_t - im * sin_t
        result.append(shifted_re)
    return result

def add_tone(x, freq, amplitude, sample_rate):
    """Add a pure tone to existing signal."""
    result = list(x)
    for i in range(len(result)):
        t = i / sample_rate
        result[i] += amplitude * math.sin(TAU * freq * t)
    return result

def crossfade(x1, x2, duration_samples):
    """Crossfade between two signals."""
    n = min(len(x1), len(x2), duration_samples)
    result = []
    for i in range(n):
        t = i / n
        result.append(x1[i] * (1 - t) + x2[i] * t)
    if len(x2) > n:
        result.extend(x2[n:])
    return result

def pan_stereo(mono, position, sample_rate=44100):
    """Pan mono signal to stereo position (-1=left, 1=right)."""
    import math
    left_gain = math.cos((position + 1) * math.pi / 4)
    right_gain = math.sin((position + 1) * math.pi / 4)
    left = [s * left_gain for s in mono]
    right = [s * right_gain for s in mono]
    return left, right

def haas_effect(mono, delay_ms=20, sample_rate=44100):
    """Create stereo image using Haas effect (precedence)."""
    delay_samples = int(delay_ms * sample_rate / 1000)
    left = list(mono)
    right = [0.0] * delay_samples + list(mono)
    min_len = min(len(left), len(right))
    return left[:min_len], right[:min_len]

def stereo_width(left, right, width=1.0):
    """Adjust stereo width. 0=mono, 1=normal, 2=extra wide."""
    mid = [(l + r) / 2 for l, r in zip(left, right)]
    side = [(l - r) / 2 for l, r in zip(left, right)]
    new_left = [m + s * width for m, s in zip(mid, side)]
    new_right = [m - s * width for m, s in zip(mid, side)]
    return new_left, new_right

def auto_gain(signals, target_rms=0.1):
    """Automatically adjust gains to match target RMS."""
    import math
    results = []
    for sig in signals:
        current_rms = math.sqrt(sum(s*s for s in sig) / max(1, len(sig)))
        if current_rms > 1e-12:
            gain = target_rms / current_rms
            results.append([s * gain for s in sig])
        else:
            results.append(list(sig))
    return results
