"""Signal normalization utilities."""
import math
from dravex.core.signal import Signal
from dravex.constants import DEFAULT_EPSILON

def peak_normalize(signal, target=1.0):
    """Normalize signal to target peak amplitude.

    Preserves waveform shape while scaling to target level.
    """
    """Normalize to target peak amplitude."""
    return signal.normalize(target)

def rms_normalize(signal, target_rms=0.1):
    """Normalize to target RMS level."""
    current_rms = signal.rms()
    if current_rms < DEFAULT_EPSILON:
        return signal
    scale = target_rms / current_rms
    return signal * scale

def lufs_normalize(signal, target_lufs=-23.0):
    """Approximate LUFS normalization (simplified ITU-R BS.1770)."""
    samples = signal.samples
    n = len(samples)
    if n == 0:
        return signal
    mean_sq = sum(s * s for s in samples) / n
    if mean_sq < DEFAULT_EPSILON:
        return signal
    current_lufs = -0.691 + 10 * math.log10(mean_sq)
    gain_db = target_lufs - current_lufs
    gain = 10.0 ** (gain_db / 20.0)
    return signal * gain

def dc_remove(signal):
    """Remove DC offset."""
    samples = signal.samples
    mean = sum(samples) / len(samples) if samples else 0
    return Signal([s - mean for s in samples], signal.sample_rate)

def pre_emphasis(signal, coeff=0.97):
    """Apply pre-emphasis filter y[n] = x[n] - coeff * x[n-1]."""
    samples = signal.samples
    result = [samples[0]]
    for i in range(1, len(samples)):
        result.append(samples[i] - coeff * samples[i - 1])
    return Signal(result, signal.sample_rate)

def de_emphasis(signal, coeff=0.97):
    """Inverse pre-emphasis: y[n] = x[n] + coeff * y[n-1]."""
    samples = signal.samples
    result = [samples[0]]
    for i in range(1, len(samples)):
        result.append(samples[i] + coeff * result[i - 1])
    return Signal(result, signal.sample_rate)

def clip(signal, min_val=-1.0, max_val=1.0):
    """Hard clip signal to range."""
    return Signal([max(min_val, min(max_val, s)) for s in signal.samples], signal.sample_rate)

def soft_clip(signal, threshold=0.8):
    """Soft clipping using tanh saturation."""
    return Signal([math.tanh(s / threshold) * threshold for s in signal.samples], signal.sample_rate)

def auto_gain_control(signal, target_rms=0.1, attack=0.01, release=0.001):
    """Automatic gain control."""
    import math
    samples = signal.samples
    n = len(samples)
    gain = 1.0
    result = []
    for i in range(n):
        out = samples[i] * gain
        result.append(max(-1, min(1, out)))
        current_rms = abs(out)
        error = target_rms - current_rms
        if error > 0:
            gain += release * error
        else:
            gain += attack * error
        gain = max(0.01, min(100.0, gain))
    return Signal(result, signal.sample_rate)

def compressor(signal, threshold_db=-20, ratio=4.0, attack_ms=5, release_ms=50):
    """Dynamic range compressor."""
    import math
    from dravex.constants import amplitude_to_db, db_to_amplitude
    samples = signal.samples
    n = len(samples)
    sr = signal.sample_rate
    attack_coeff = math.exp(-1.0 / (attack_ms * sr / 1000)) if attack_ms > 0 else 0
    release_coeff = math.exp(-1.0 / (release_ms * sr / 1000)) if release_ms > 0 else 0
    envelope = 0.0
    result = []
    for s in samples:
        level = abs(s)
        if level > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * level
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * level
        level_db = amplitude_to_db(max(envelope, 1e-12))
        if level_db > threshold_db:
            gain_reduction = (level_db - threshold_db) * (1 - 1/ratio)
            gain = db_to_amplitude(-gain_reduction)
        else:
            gain = 1.0
        result.append(s * gain)
    return Signal(result, signal.sample_rate)

def limiter(signal, threshold=0.95):
    """Hard limiter (brick wall)."""
    samples = signal.samples
    result = []
    for s in samples:
        if abs(s) > threshold:
            result.append(threshold if s > 0 else -threshold)
        else:
            result.append(s)
    return Signal(result, signal.sample_rate)

def noise_gate(signal, threshold_db=-40, attack_ms=1, release_ms=50):
    """Noise gate."""
    import math
    from dravex.constants import amplitude_to_db
    samples = signal.samples
    n = len(samples)
    sr = signal.sample_rate
    attack_coeff = math.exp(-1.0 / (attack_ms * sr / 1000)) if attack_ms > 0 else 0
    release_coeff = math.exp(-1.0 / (release_ms * sr / 1000)) if release_ms > 0 else 0
    gate = 0.0
    result = []
    for s in samples:
        level_db = amplitude_to_db(max(abs(s), 1e-12))
        target = 1.0 if level_db >= threshold_db else 0.0
        if target > gate:
            gate = attack_coeff * gate + (1 - attack_coeff) * target
        else:
            gate = release_coeff * gate + (1 - release_coeff) * target
        result.append(s * gate)
    return Signal(result, signal.sample_rate)

def fade_envelope(signal, attack_ms=10, decay_ms=50, sustain_level=0.7, release_ms=100):
    """Apply ADSR-like fade envelope to signal."""
    import math
    samples = signal.samples
    n = len(samples)
    sr = signal.sample_rate
    attack = int(attack_ms * sr / 1000)
    decay = int(decay_ms * sr / 1000)
    release = int(release_ms * sr / 1000)
    sustain_end = n - release
    result = []
    for i in range(n):
        if i < attack:
            env = i / attack if attack > 0 else 1.0
        elif i < attack + decay:
            t = (i - attack) / decay if decay > 0 else 1.0
            env = 1.0 - (1.0 - sustain_level) * t
        elif i < sustain_end:
            env = sustain_level
        else:
            t = (i - sustain_end) / release if release > 0 else 0
            env = sustain_level * (1.0 - t)
        result.append(samples[i] * max(0, env))
    return Signal(result, sr)
