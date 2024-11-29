"""Goertzel algorithm for single-frequency DFT."""
import math
from dravex.constants import TAU

def goertzel(x, target_freq, sample_rate):
    """Goertzel algorithm: DFT at a single frequency. O(N)."""
    N = len(x)
    k = int(round(target_freq * N / sample_rate))
    w = TAU * k / N
    coeff = 2 * math.cos(w)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0
    real = s1 - s2 * math.cos(w)
    imag = s2 * math.sin(w)
    return math.sqrt(real * real + imag * imag), math.atan2(imag, real)

def goertzel_magnitude(x, target_freq, sample_rate):
    """Magnitude at a single frequency."""
    mag, _ = goertzel(x, target_freq, sample_rate)
    return mag

def multi_goertzel(x, frequencies, sample_rate):
    """Goertzel for multiple frequencies."""
    return [goertzel(x, f, sample_rate) for f in frequencies]

def dtmf_detect(x, sample_rate):
    """Detect DTMF tones using Goertzel."""
    low_freqs = [697, 770, 852, 941]
    high_freqs = [1209, 1336, 1477, 1633]
    keypad = {
        (697,1209):'1',(697,1336):'2',(697,1477):'3',(697,1633):'A',
        (770,1209):'4',(770,1336):'5',(770,1477):'6',(770,1633):'B',
        (852,1209):'7',(852,1336):'8',(852,1477):'9',(852,1633):'C',
        (941,1209):'*',(941,1336):'0',(941,1477):'#',(941,1633):'D',
    }
    low_mags = {f: goertzel_magnitude(x, f, sample_rate) for f in low_freqs}
    high_mags = {f: goertzel_magnitude(x, f, sample_rate) for f in high_freqs}
    best_low = max(low_freqs, key=lambda f: low_mags[f])
    best_high = max(high_freqs, key=lambda f: high_mags[f])
    threshold = max(max(low_mags.values()), max(high_mags.values())) * 0.3
    if low_mags[best_low] < threshold or high_mags[best_high] < threshold:
        return None
    return keypad.get((best_low, best_high))

def goertzel_power(x, target_freq, sample_rate):
    """Power at target frequency."""
    mag, _ = goertzel(x, target_freq, sample_rate)
    return mag * mag

def sliding_goertzel(x, target_freq, sample_rate, frame_size, hop_size):
    """Compute Goertzel magnitude over sliding windows."""
    n = len(x)
    results = []
    pos = 0
    while pos + frame_size <= n:
        frame = x[pos:pos + frame_size]
        mag = goertzel_magnitude(frame, target_freq, sample_rate)
        results.append(mag)
        pos += hop_size
    return results

def tone_detect(x, frequencies, sample_rate, threshold=0.5):
    """Detect which tones are present above threshold."""
    max_mag = max(goertzel_magnitude(x, f, sample_rate) for f in frequencies) or 1.0
    detected = []
    for f in frequencies:
        mag = goertzel_magnitude(x, f, sample_rate)
        if mag / max_mag > threshold:
            detected.append(f)
    return detected
