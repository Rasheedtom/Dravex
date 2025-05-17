"""Matched filter for signal detection in noise."""
import math
from dravex.constants import DEFAULT_EPSILON

def matched_filter(signal, template):
    """Apply matched filter: correlate signal with time-reversed template."""
    h = list(reversed(template))
    n = len(signal)
    m = len(h)
    output = []
    for i in range(n):
        s = 0.0
        for j in range(m):
            idx = i - j
            if 0 <= idx < n:
                s += signal[idx] * h[j]
        output.append(s)
    return output

def detect_template(signal, template, threshold=0.7):
    """Detect occurrences of template in signal."""
    mf = matched_filter(signal, template)
    energy = math.sqrt(sum(t * t for t in template))
    if energy < DEFAULT_EPSILON:
        return []
    normalized = [m / (energy * energy) for m in mf]
    detections = []
    n = len(normalized)
    for i in range(1, n - 1):
        if normalized[i] > threshold:
            if normalized[i] >= normalized[i-1] and normalized[i] >= normalized[i+1]:
                detections.append((i, normalized[i]))
    return detections

def snr_gain(template_length):
    """SNR improvement from matched filtering."""
    return 10 * math.log10(template_length) if template_length > 0 else 0.0

def pulse_compress(signal, chirp_template):
    """Pulse compression via matched filtering with chirp."""
    return matched_filter(signal, chirp_template)
