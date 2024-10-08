"""Signal segmentation utilities."""
import math
from dravex.core.signal import Signal

def segment_fixed(signal, segment_length, overlap=0):
    """Split signal into fixed-length segments."""
    samples = signal.samples
    n = len(samples)
    step = segment_length - overlap
    if step <= 0:
        step = 1
    segments = []
    pos = 0
    while pos + segment_length <= n:
        seg = Signal(samples[pos:pos + segment_length], signal.sample_rate)
        segments.append(seg)
        pos += step
    return segments

def segment_by_silence(signal, threshold=0.01, min_silence=100):
    """Split signal at silence regions.

    Silence is defined as consecutive samples below threshold amplitude.
    min_silence controls minimum gap length in samples.
    """
    """Split signal at silence regions."""
    samples = signal.samples
    n = len(samples)
    segments = []
    in_silence = False
    silence_start = 0
    seg_start = 0
    for i in range(n):
        is_silent = abs(samples[i]) < threshold
        if is_silent and not in_silence:
            in_silence = True
            silence_start = i
        elif not is_silent and in_silence:
            if i - silence_start >= min_silence:
                if silence_start > seg_start:
                    segments.append(Signal(samples[seg_start:silence_start], signal.sample_rate))
                seg_start = i
            in_silence = False
    if seg_start < n:
        segments.append(Signal(samples[seg_start:], signal.sample_rate))
    return segments

def segment_by_energy(signal, frame_size=1024, hop_size=512, threshold_db=-40):
    """Split signal by energy threshold."""
    from dravex.constants import power_to_db
    samples = signal.samples
    n = len(samples)
    active = []
    pos = 0
    while pos + frame_size <= n:
        frame = samples[pos:pos + frame_size]
        energy = sum(s * s for s in frame) / frame_size
        db = power_to_db(energy)
        active.append(db >= threshold_db)
        pos += hop_size
    segments = []
    seg_start = None
    for i, is_active in enumerate(active):
        if is_active and seg_start is None:
            seg_start = i * hop_size
        elif not is_active and seg_start is not None:
            end = i * hop_size
            segments.append(Signal(samples[seg_start:end], signal.sample_rate))
            seg_start = None
    if seg_start is not None:
        segments.append(Signal(samples[seg_start:], signal.sample_rate))
    return segments

def concatenate_signals(signals, crossfade=0):
    """Concatenate multiple signals with optional crossfade."""
    if not signals:
        return Signal([], 44100)
    result = list(signals[0].samples)
    sr = signals[0].sample_rate
    for sig in signals[1:]:
        new = sig.samples
        if crossfade > 0 and len(result) >= crossfade and len(new) >= crossfade:
            for i in range(crossfade):
                t = i / crossfade
                result[len(result) - crossfade + i] = result[len(result) - crossfade + i] * (1 - t) + new[i] * t
            result.extend(new[crossfade:])
        else:
            result.extend(new)
    return Signal(result, sr)

def sliding_window(signal, window_size, hop_size=1):
    """Generate sliding windows over signal."""
    samples = signal.samples if hasattr(signal, 'samples') else list(signal)
    n = len(samples)
    windows = []
    pos = 0
    while pos + window_size <= n:
        windows.append(samples[pos:pos + window_size])
        pos += hop_size
    return windows

def zero_crossing_segments(signal, min_segment=10):
    """Segment signal at zero crossings."""
    samples = signal.samples if hasattr(signal, 'samples') else list(signal)
    n = len(samples)
    segments = []
    seg_start = 0
    for i in range(1, n):
        if samples[i-1] * samples[i] < 0 and i - seg_start >= min_segment:
            from dravex.core.signal import Signal
            sr = signal.sample_rate if hasattr(signal, 'sample_rate') else 44100
            segments.append(Signal(samples[seg_start:i], sr))
            seg_start = i
    if seg_start < n:
        sr = signal.sample_rate if hasattr(signal, 'sample_rate') else 44100
        segments.append(Signal(samples[seg_start:], sr))
    return segments

def adaptive_segmentation(signal, min_size=256, max_size=4096, threshold=2.0):
    """Adaptive segmentation based on spectral change."""
    import math
    samples = signal.samples if hasattr(signal, 'samples') else list(signal)
    n = len(samples)
    sr = signal.sample_rate if hasattr(signal, 'sample_rate') else 44100
    boundaries = [0]
    pos = min_size
    while pos < n - min_size:
        left = samples[max(0, pos-min_size):pos]
        right = samples[pos:min(n, pos+min_size)]
        left_e = sum(s*s for s in left) / len(left) if left else 0
        right_e = sum(s*s for s in right) / len(right) if right else 0
        ratio = abs(left_e - right_e) / max(min(left_e, right_e), 1e-12)
        if ratio > threshold and pos - boundaries[-1] >= min_size:
            boundaries.append(pos)
        pos += min_size // 2
    boundaries.append(n)
    from dravex.core.signal import Signal
    segments = []
    for i in range(len(boundaries) - 1):
        segments.append(Signal(samples[boundaries[i]:boundaries[i+1]], sr))
    return segments
