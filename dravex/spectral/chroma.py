"""Chroma feature extraction."""
import math
from dravex.transform.fft import rfft
from dravex.constants import DEFAULT_EPSILON

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def chroma_vector(magnitudes, sample_rate, n_fft):
    """Compute 12-bin chroma vector from magnitude spectrum."""
    chroma = [0.0] * 12
    for k in range(1, len(magnitudes)):
        freq = k * sample_rate / n_fft
        if freq < 20 or freq > 5000:
            continue
        midi = 69 + 12 * math.log2(freq / 440.0) if freq > 0 else 0
        pitch_class = int(round(midi)) % 12
        chroma[pitch_class] += magnitudes[k] ** 2
    total = sum(chroma)
    if total > DEFAULT_EPSILON:
        chroma = [c / total for c in chroma]
    return chroma

def chromagram(spectrogram_frames, sample_rate, n_fft):
    """Compute chromagram from spectrogram frames."""
    return [chroma_vector(frame, sample_rate, n_fft) for frame in spectrogram_frames]

def chord_template(root, quality="major"):
    """Generate chord template for matching."""
    template = [0.0] * 12
    if quality == "major":
        intervals = [0, 4, 7]
    elif quality == "minor":
        intervals = [0, 3, 7]
    elif quality == "dim":
        intervals = [0, 3, 6]
    else:
        intervals = [0, 4, 7]
    for i in intervals:
        template[(root + i) % 12] = 1.0
    norm = math.sqrt(sum(t * t for t in template))
    if norm > DEFAULT_EPSILON:
        template = [t / norm for t in template]
    return template

def detect_chord(chroma):
    """Detect chord from chroma vector via template matching."""
    best_match = ""
    best_score = -1
    for root in range(12):
        for quality in ["major", "minor"]:
            tmpl = chord_template(root, quality)
            score = sum(c * t for c, t in zip(chroma, tmpl))
            if score > best_score:
                best_score = score
                suffix = "" if quality == "major" else "m"
                best_match = NOTE_NAMES[root] + suffix
    return best_match, best_score

def key_detection(chromagram):
    """Estimate musical key from chromagram."""
    avg_chroma = [0.0] * 12
    for frame in chromagram:
        for i in range(12):
            avg_chroma[i] += frame[i]
    n = len(chromagram)
    if n > 0:
        avg_chroma = [c / n for c in avg_chroma]
    return detect_chord(avg_chroma)

def pitch_class_histogram(magnitudes, sample_rate, n_fft):
    """Histogram of pitch classes weighted by magnitude."""
    import math
    hist = [0.0] * 12
    for k in range(1, len(magnitudes)):
        freq = k * sample_rate / n_fft
        if freq < 20 or freq > 5000:
            continue
        midi = 69 + 12 * math.log2(freq / 440.0) if freq > 0 else 0
        pc = int(round(midi)) % 12
        hist[pc] += magnitudes[k]
    return hist

def harmonic_change_detection(chromagram):
    """Detect harmonic changes between consecutive chroma frames."""
    import math
    changes = [0.0]
    for i in range(1, len(chromagram)):
        dot = sum(a * b for a, b in zip(chromagram[i], chromagram[i-1]))
        na = math.sqrt(sum(a*a for a in chromagram[i]))
        nb = math.sqrt(sum(b*b for b in chromagram[i-1]))
        if na > 1e-12 and nb > 1e-12:
            sim = dot / (na * nb)
            changes.append(1.0 - max(0, min(1, sim)))
        else:
            changes.append(0.0)
    return changes

def tonnetz(chroma):
    """Compute Tonnetz (tonal centroid) features from chroma."""
    import math
    phi = [2 * math.pi * i / 12 for i in range(12)]
    features = [0.0] * 6
    for i in range(12):
        features[0] += chroma[i] * math.sin(phi[i] * 7 / 12)
        features[1] += chroma[i] * math.cos(phi[i] * 7 / 12)
        features[2] += chroma[i] * math.sin(phi[i] * 3 / 12)
        features[3] += chroma[i] * math.cos(phi[i] * 3 / 12)
        features[4] += chroma[i] * math.sin(phi[i] * 4 / 12)
        features[5] += chroma[i] * math.cos(phi[i] * 4 / 12)
    return features
