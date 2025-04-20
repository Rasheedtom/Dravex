"""Frequency modulation and demodulation."""
import math
from dravex.constants import TAU, DEFAULT_EPSILON

def fm_modulate(signal, carrier_freq, modulation_index, sample_rate):
    """FM modulation: y = cos(2*pi*fc*t + beta * integral(x)).

    The modulation index (beta) determines the frequency
    deviation range and bandwidth of the FM signal.
    """
    """FM modulation: y = cos(2*pi*fc*t + beta * integral(x))."""
    n = len(signal)
    result = []
    phase = 0.0
    for i in range(n):
        t = i / sample_rate
        phase += TAU * carrier_freq / sample_rate + modulation_index * signal[i] / sample_rate
        if phase > 1e6:
            phase -= TAU * int(phase / TAU)
        result.append(math.cos(phase))
    return result

def fm_demodulate(modulated, sample_rate):
    """FM demodulation via instantaneous frequency."""
    from dravex.transform.hilbert import instantaneous_frequency
    return instantaneous_frequency(modulated, sample_rate)

def vibrato(signal, rate, depth_samples, sample_rate):
    """Vibrato effect (frequency modulation at sub-audio rate)."""
    n = len(signal)
    result = [0.0] * n
    for i in range(n):
        t = i / sample_rate
        delay = depth_samples * math.sin(TAU * rate * t)
        idx = i - delay
        int_idx = int(idx)
        frac = idx - int_idx
        if 0 <= int_idx < n - 1:
            result[i] = signal[int_idx] * (1 - frac) + signal[int_idx + 1] * frac
        elif 0 <= int_idx < n:
            result[i] = signal[int_idx]
    return result

def chirp(duration, start_freq, end_freq, sample_rate=44100):
    """Generate linear frequency sweep (chirp)."""
    n = int(duration * sample_rate)
    result = []
    for i in range(n):
        t = i / sample_rate
        freq = start_freq + (end_freq - start_freq) * t / duration
        phase = TAU * (start_freq * t + (end_freq - start_freq) * t * t / (2 * duration))
        result.append(math.sin(phase))
    return result
