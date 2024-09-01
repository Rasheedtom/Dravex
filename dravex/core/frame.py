"""Windowed signal frame for spectral analysis."""
import math
from dravex.core.signal import Signal

class Frame:
    """A single windowed frame of signal data."""

    def __init__(self, samples, window=None, sample_rate=44100, center_time=0.0):
        self._samples = list(samples)
        self._window = window
        self._sample_rate = sample_rate
        self._center_time = center_time
        if window is not None and len(window) == len(self._samples):
            self._windowed = [s * w for s, w in zip(self._samples, window)]
        else:
            self._windowed = list(self._samples)

    @property
    def raw_samples(self):
        return list(self._samples)

    @property
    def windowed_samples(self):
        return list(self._windowed)

    @property
    def size(self):
        return len(self._samples)

    @property
    def center_time(self):
        return self._center_time

    def energy(self):
        return sum(s * s for s in self._windowed)

    def rms(self):
        if not self._windowed:
            return 0.0
        return math.sqrt(sum(s * s for s in self._windowed) / len(self._windowed))

    def zero_crossings(self):
        count = 0
        for i in range(1, len(self._windowed)):
            if self._windowed[i-1] * self._windowed[i] < 0:
                count += 1
        return count

    def autocorrelation(self, max_lag=None):
        n = len(self._windowed)
        if max_lag is None:
            max_lag = n
        result = []
        for lag in range(min(max_lag, n)):
            s = 0.0
            for i in range(n - lag):
                s += self._windowed[i] * self._windowed[i + lag]
            result.append(s)
        return result

    def spectral_centroid(self):
        from dravex.transform.fft import rfft
        magnitudes = rfft(self._windowed)
        total = sum(magnitudes)
        if total < 1e-12:
            return 0.0
        weighted = sum(i * m for i, m in enumerate(magnitudes))
        bin_hz = self._sample_rate / len(self._windowed)
        return (weighted / total) * bin_hz


def extract_frames(signal, frame_size, hop_size, window=None):
    """Extract overlapping frames from a signal."""
    samples = signal.samples if isinstance(signal, Signal) else list(signal)
    sr = signal.sample_rate if isinstance(signal, Signal) else 44100
    frames = []
    n = len(samples)
    pos = 0
    while pos + frame_size <= n:
        chunk = samples[pos:pos + frame_size]
        t = (pos + frame_size / 2) / sr
        frames.append(Frame(chunk, window=window, sample_rate=sr, center_time=t))
        pos += hop_size
    return frames

def overlap_frames(signal, frame_size, hop_size, window=None):
    """Create overlapping frames with synthesis window."""
    from dravex.core.signal import Signal
    samples = signal.samples if isinstance(signal, Signal) else list(signal)
    sr = signal.sample_rate if isinstance(signal, Signal) else 44100
    n = len(samples)
    frames = []
    positions = []
    pos = 0
    while pos + frame_size <= n:
        chunk = samples[pos:pos + frame_size]
        if window is not None:
            chunk = [s * w for s, w in zip(chunk, window)]
        frames.append(chunk)
        positions.append(pos)
        pos += hop_size
    return frames, positions

def reconstruct_from_frames(frames, positions, frame_size, window=None):
    """Reconstruct signal from overlapping frames (overlap-add)."""
    if not frames:
        return []
    total_len = positions[-1] + frame_size
    output = [0.0] * total_len
    norm = [0.0] * total_len
    for frame, pos in zip(frames, positions):
        for i in range(len(frame)):
            if window is not None:
                w = window[i] if i < len(window) else 1.0
                output[pos + i] += frame[i] * w
                norm[pos + i] += w * w
            else:
                output[pos + i] += frame[i]
                norm[pos + i] += 1.0
    for i in range(total_len):
        if norm[i] > 1e-8:
            output[i] /= norm[i]
    return output

def frame_features(frame):
    """Compute standard frame-level features."""
    import math
    n = len(frame)
    if n == 0:
        return {}
    energy = sum(s * s for s in frame) / n
    rms_val = math.sqrt(energy)
    zcr = sum(1 for i in range(1, n) if frame[i-1] * frame[i] < 0) / (n - 1)
    mean_val = sum(frame) / n
    peak = max(abs(s) for s in frame)
    return {
        "energy": energy,
        "rms": rms_val,
        "zcr": zcr,
        "mean": mean_val,
        "peak": peak,
        "crest_factor": peak / rms_val if rms_val > 1e-12 else 0,
    }
