"""Tone and test signal generation."""
import math
from dravex.constants import TAU

def sine_wave(freq, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    """Generate sine wave."""
    n = int(duration * sample_rate)
    return [amplitude * math.sin(TAU * freq * i / sample_rate + phase) for i in range(n)]

def square_wave(freq, duration, sample_rate=44100, amplitude=1.0, duty=0.5):
    """Generate square wave with variable duty cycle."""
    n = int(duration * sample_rate)
    period = sample_rate / freq
    result = []
    for i in range(n):
        phase = (i % period) / period
        result.append(amplitude if phase < duty else -amplitude)
    return result

def sawtooth_wave(freq, duration, sample_rate=44100, amplitude=1.0):
    """Generate sawtooth wave."""
    n = int(duration * sample_rate)
    period = sample_rate / freq
    return [amplitude * (2 * ((i / period) % 1.0) - 1) for i in range(n)]

def triangle_wave(freq, duration, sample_rate=44100, amplitude=1.0):
    """Generate triangle wave."""
    n = int(duration * sample_rate)
    period = sample_rate / freq
    result = []
    for i in range(n):
        phase = (i / period) % 1.0
        if phase < 0.25:
            result.append(amplitude * 4 * phase)
        elif phase < 0.75:
            result.append(amplitude * (2 - 4 * phase))
        else:
            result.append(amplitude * (4 * phase - 4))
    return result

def impulse(length, position=0, amplitude=1.0, sample_rate=44100):
    """Generate impulse (Dirac delta approximation)."""
    result = [0.0] * length
    if 0 <= position < length:
        result[position] = amplitude
    return result

def sweep(start_freq, end_freq, duration, sample_rate=44100):
    """Generate logarithmic frequency sweep."""
    n = int(duration * sample_rate)
    result = []
    for i in range(n):
        t = i / sample_rate
        freq = start_freq * (end_freq / start_freq) ** (t / duration)
        phase = TAU * start_freq * duration / math.log(end_freq / start_freq) * ((end_freq/start_freq)**(t/duration) - 1)
        result.append(math.sin(phase))
    return result

def multi_tone(frequencies, duration, sample_rate=44100, amplitudes=None):
    """Generate sum of multiple sine waves."""
    import math
    n = int(duration * sample_rate)
    if amplitudes is None:
        amplitudes = [1.0 / len(frequencies)] * len(frequencies)
    result = [0.0] * n
    for freq, amp in zip(frequencies, amplitudes):
        for i in range(n):
            result[i] += amp * math.sin(2 * math.pi * freq * i / sample_rate)
    return result

def dtmf_tone(digit, duration=0.1, sample_rate=44100):
    """Generate DTMF tone for a digit."""
    import math
    dtmf_map = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477),
    }
    freqs = dtmf_map.get(str(digit), (697, 1209))
    n = int(duration * sample_rate)
    result = []
    for i in range(n):
        t = i / sample_rate
        result.append(0.5 * math.sin(2*math.pi*freqs[0]*t) + 0.5 * math.sin(2*math.pi*freqs[1]*t))
    return result

def white_noise_burst(duration, amplitude=1.0, sample_rate=44100, rng=None):
    """Generate a burst of white noise."""
    import random as _r
    if rng is None:
        rng = _r.Random(42)
    n = int(duration * sample_rate)
    return [amplitude * (2 * rng.random() - 1) for _ in range(n)]

def am_tone(carrier_freq, mod_freq, duration, sample_rate=44100, depth=0.5):
    """Generate AM-modulated tone."""
    import math
    n = int(duration * sample_rate)
    result = []
    for i in range(n):
        t = i / sample_rate
        carrier = math.sin(2 * math.pi * carrier_freq * t)
        modulator = 1.0 + depth * math.sin(2 * math.pi * mod_freq * t)
        result.append(carrier * modulator)
    return result

def fm_tone(carrier_freq, mod_freq, mod_index, duration, sample_rate=44100):
    """Generate FM-modulated tone."""
    import math
    n = int(duration * sample_rate)
    result = []
    phase = 0.0
    for i in range(n):
        t = i / sample_rate
        phase += 2 * math.pi * (carrier_freq + mod_index * math.sin(2 * math.pi * mod_freq * t)) / sample_rate
        result.append(math.sin(phase))
    return result

def harmonic_series(fundamental, n_harmonics, duration, sample_rate=44100, decay=0.7):
    """Generate harmonic series with exponential amplitude decay."""
    import math
    n = int(duration * sample_rate)
    result = [0.0] * n
    for h in range(1, n_harmonics + 1):
        amp = decay ** (h - 1)
        freq = fundamental * h
        if freq >= sample_rate / 2:
            break
        for i in range(n):
            result[i] += amp * math.sin(2 * math.pi * freq * i / sample_rate)
    peak = max(abs(s) for s in result) or 1.0
    return [s / peak for s in result]

def click_track(bpm, duration, sample_rate=44100):
    """Generate metronome click track."""
    import math
    n = int(duration * sample_rate)
    result = [0.0] * n
    samples_per_beat = int(60 * sample_rate / bpm)
    click_len = min(200, samples_per_beat // 4)
    for beat_start in range(0, n, samples_per_beat):
        for i in range(click_len):
            if beat_start + i < n:
                t = i / click_len
                result[beat_start + i] = math.sin(2*math.pi*1000*i/sample_rate) * (1 - t)
    return result
