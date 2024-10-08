"""Signal envelope extraction."""
import math
from dravex.core.signal import Signal
from dravex.constants import DEFAULT_EPSILON

def amplitude_envelope(signal, frame_size=1024, hop_size=512):
    """Extract amplitude envelope (peak per frame)."""
    samples = signal.samples
    n = len(samples)
    envelope = []
    pos = 0
    while pos + frame_size <= n:
        peak = max(abs(s) for s in samples[pos:pos + frame_size])
        envelope.append(peak)
        pos += hop_size
    return envelope

def rms_envelope(signal, frame_size=1024, hop_size=512):
    """Extract RMS envelope."""
    samples = signal.samples
    n = len(samples)
    envelope = []
    pos = 0
    while pos + frame_size <= n:
        frame = samples[pos:pos + frame_size]
        rms_val = math.sqrt(sum(s * s for s in frame) / len(frame))
        envelope.append(rms_val)
        pos += hop_size
    return envelope

def hilbert_envelope(signal):
    """Analytic signal envelope via Hilbert transform."""
    from dravex.transform.hilbert import hilbert_transform
    analytic = hilbert_transform(signal.samples)
    return [abs(complex(r, i)) for r, i in zip(signal.samples, analytic)]

def attack_decay_sustain_release(envelope, threshold=0.1):
    """Estimate ADSR segments from an amplitude envelope.

    Returns dict with attack, decay, sustain, release sample counts.
    """
    """Estimate ADSR segments from an envelope."""
    n = len(envelope)
    if n == 0:
        return {"attack": 0, "decay": 0, "sustain": 0, "release": 0}
    peak_idx = max(range(n), key=lambda i: envelope[i])
    peak_val = envelope[peak_idx]
    if peak_val < DEFAULT_EPSILON:
        return {"attack": n, "decay": 0, "sustain": 0, "release": 0}
    attack_end = peak_idx
    sustain_level = peak_val * 0.7
    decay_end = attack_end
    for i in range(attack_end, n):
        if envelope[i] <= sustain_level + DEFAULT_EPSILON:
            decay_end = i
            break
    release_start = n
    for i in range(n - 1, decay_end - 1, -1):
        if envelope[i] > threshold * peak_val:
            release_start = i
            break
    return {
        "attack": attack_end,
        "decay": decay_end - attack_end,
        "sustain": release_start - decay_end,
        "release": n - release_start,
    }

def smooth_envelope(envelope, window_size=5):
    """Smooth envelope using moving average."""
    n = len(envelope)
    half = window_size // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(envelope[lo:hi]) / (hi - lo))
    return result

def peak_envelope(signal, attack_coeff=0.01, release_coeff=0.001):
    """Peak-following envelope detector."""
    samples = signal.samples if hasattr(signal, 'samples') else list(signal)
    env = [0.0]
    for i in range(1, len(samples)):
        amp = abs(samples[i])
        if amp > env[-1]:
            env.append(env[-1] + attack_coeff * (amp - env[-1]))
        else:
            env.append(env[-1] + release_coeff * (amp - env[-1]))
    return env

def temporal_centroid(envelope):
    """Temporal centroid of an envelope (center of gravity in time)."""
    total = sum(envelope)
    if total < 1e-12:
        return 0.0
    weighted = sum(i * e for i, e in enumerate(envelope))
    return weighted / total

def attack_time(envelope, threshold=0.9):
    """Time to reach threshold fraction of peak."""
    if not envelope:
        return 0
    peak = max(envelope)
    target = peak * threshold
    for i, e in enumerate(envelope):
        if e >= target:
            return i
    return len(envelope)

def decay_time(envelope, threshold=0.1):
    """Time from peak to fall below threshold fraction."""
    if not envelope:
        return 0
    peak_idx = max(range(len(envelope)), key=lambda i: envelope[i])
    peak = envelope[peak_idx]
    target = peak * threshold
    for i in range(peak_idx, len(envelope)):
        if envelope[i] < target:
            return i - peak_idx
    return len(envelope) - peak_idx

def envelope_follower(signal, attack_ms=10, release_ms=100, sample_rate=44100):
    """Analog-style envelope follower."""
    import math
    samples = signal.samples if hasattr(signal, 'samples') else list(signal)
    attack = math.exp(-1.0 / (attack_ms * sample_rate / 1000))
    release = math.exp(-1.0 / (release_ms * sample_rate / 1000))
    env = [0.0]
    for i in range(1, len(samples)):
        amp = abs(samples[i])
        if amp > env[-1]:
            env.append(attack * env[-1] + (1 - attack) * amp)
        else:
            env.append(release * env[-1] + (1 - release) * amp)
    return env
