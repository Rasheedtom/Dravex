"""Amplitude modulation and demodulation."""
import math
from dravex.constants import TAU

def am_modulate(signal, carrier_freq, sample_rate, depth=1.0):
    """AM modulation: y = (1 + m*x) * cos(2*pi*fc*t).

    Depth of 1.0 gives 100% modulation; values above 1.0
    result in over-modulation with envelope inversion.
    """
    """AM modulation: y = (1 + m*x) * cos(2*pi*fc*t)."""
    n = len(signal)
    result = []
    for i in range(n):
        t = i / sample_rate
        carrier = math.cos(TAU * carrier_freq * t)
        result.append((1 + depth * signal[i]) * carrier)
    return result

def am_demodulate(modulated, carrier_freq, sample_rate, cutoff=None):
    """AM demodulation via envelope detection."""
    if cutoff is None:
        cutoff = carrier_freq / 4
    n = len(modulated)
    rectified = [abs(s) for s in modulated]
    from dravex.filter.fir import fir_lowpass, apply_fir
    h = fir_lowpass(min(64, n // 4), cutoff, sample_rate)
    demod = apply_fir(rectified, h)
    dc = sum(demod) / len(demod) if demod else 0
    return [s - dc for s in demod]

def dsb_sc_modulate(signal, carrier_freq, sample_rate):
    """Double-sideband suppressed-carrier modulation."""
    result = []
    for i in range(len(signal)):
        t = i / sample_rate
        result.append(signal[i] * math.cos(TAU * carrier_freq * t))
    return result

def ring_modulate(signal1, signal2):
    """Ring modulation: element-wise multiplication."""
    n = min(len(signal1), len(signal2))
    return [signal1[i] * signal2[i] for i in range(n)]

def tremolo(signal, rate, depth, sample_rate):
    """Tremolo effect (amplitude modulation at sub-audio rate)."""
    result = []
    for i in range(len(signal)):
        t = i / sample_rate
        mod = 1.0 - depth * (0.5 + 0.5 * math.sin(TAU * rate * t))
        result.append(signal[i] * mod)
    return result

def ssb_modulate(signal, carrier_freq, sample_rate, upper=True):
    """Single-sideband modulation."""
    import math
    from dravex.transform.hilbert import hilbert_transform
    ht = hilbert_transform(signal)
    result = []
    for i in range(len(signal)):
        t = i / sample_rate
        cos_t = math.cos(2 * math.pi * carrier_freq * t)
        sin_t = math.sin(2 * math.pi * carrier_freq * t)
        if upper:
            result.append(signal[i] * cos_t - ht[i] * sin_t)
        else:
            result.append(signal[i] * cos_t + ht[i] * sin_t)
    return result

def am_modulation_index(modulated, carrier_freq, sample_rate):
    """Estimate AM modulation index from modulated signal."""
    envelope = [abs(s) for s in modulated]
    n = len(envelope)
    if n < 100:
        return 0.0
    window = max(1, int(sample_rate / carrier_freq * 2))
    smoothed = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window)
        smoothed.append(sum(envelope[lo:hi]) / (hi - lo))
    peak = max(smoothed)
    trough = min(s for s in smoothed if s > 0.01 * peak)
    if peak + trough < 1e-10:
        return 0.0
    return (peak - trough) / (peak + trough)
