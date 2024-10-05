"""1D discrete signal container."""
import math
from dravex.constants import DEFAULT_EPSILON, rms, amplitude_to_db
from dravex.errors import EmptySignalError, DimensionError

class Signal:
    """Immutable 1D discrete-time signal."""
    __slots__ = ("_samples", "_sample_rate")

    def __init__(self, samples, sample_rate=44100):
        if sample_rate <= 0:
            sample_rate = 44100
        self._samples = tuple(float(s) for s in samples)
        self._sample_rate = max(1, int(sample_rate))

    @property
    def samples(self):
        return list(self._samples)

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def length(self):
        return len(self._samples)

    @property
    def duration(self):
        if self._sample_rate <= 0:
            return 0.0
        return self.length / self._sample_rate

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Signal(self._samples[idx], self._sample_rate)
        return self._samples[idx]

    def __add__(self, other):
        if isinstance(other, Signal):
            if self.length != other.length:
                raise DimensionError(self.length, other.length)
            return Signal([a + b for a, b in zip(self._samples, other._samples)], self._sample_rate)
        return Signal([s + other for s in self._samples], self._sample_rate)

    def __mul__(self, scalar):
        return Signal([s * scalar for s in self._samples], self._sample_rate)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return Signal([-s for s in self._samples], self._sample_rate)

    def max_amplitude(self):
        if not self._samples:
            raise EmptySignalError()
        return max(abs(s) for s in self._samples)

    def rms(self):
        return rms(self._samples)

    def rms_db(self):
        r = self.rms()
        return amplitude_to_db(r) if r > 0 else -120.0

    def energy(self):
        return sum(s * s for s in self._samples)

    def normalize(self, target=1.0):
        peak = self.max_amplitude()
        if peak < DEFAULT_EPSILON:
            return Signal(self._samples, self._sample_rate)
        scale = target / peak
        return self * scale

    def fade_in(self, n_samples):
        n = min(n_samples, self.length)
        result = list(self._samples)
        for i in range(n):
            result[i] *= i / n
        return Signal(result, self._sample_rate)

    def fade_out(self, n_samples):
        n = min(n_samples, self.length)
        result = list(self._samples)
        for i in range(n):
            result[self.length - 1 - i] *= i / n
        return Signal(result, self._sample_rate)

    def reverse(self):
        return Signal(list(reversed(self._samples)), self._sample_rate)

    def pad_zeros(self, total_length):
        if total_length <= self.length:
            return Signal(self._samples, self._sample_rate)
        padding = [0.0] * (total_length - self.length)
        return Signal(list(self._samples) + padding, self._sample_rate)

    def trim_silence(self, threshold=0.001):
        start = 0
        while start < self.length and abs(self._samples[start]) < threshold:
            start += 1
        end = self.length
        while end > start and abs(self._samples[end - 1]) < threshold:
            end -= 1
        return Signal(self._samples[start:end], self._sample_rate)

    def concatenate(self, other):
        return Signal(list(self._samples) + list(other._samples), self._sample_rate)

    @staticmethod
    def zeros(length, sample_rate=44100):
        return Signal([0.0] * length, sample_rate)

    @staticmethod
    def ones(length, sample_rate=44100):
        return Signal([1.0] * length, sample_rate)

    def apply(self, func):
        """Apply element-wise function to samples."""
        return Signal([func(s) for s in self._samples], self._sample_rate)

    def downsample(self, factor):
        """Simple decimation without filtering."""
        return Signal(self._samples[::factor], self._sample_rate // factor)

    def upsample(self, factor):
        """Zero-insertion upsampling."""
        result = []
        for s in self._samples:
            result.append(s)
            result.extend([0.0] * (factor - 1))
        return Signal(result, self._sample_rate * factor)

    def split(self, n_segments):
        """Split signal into n equal segments."""
        seg_len = self.length // n_segments
        segments = []
        for i in range(n_segments):
            start = i * seg_len
            end = start + seg_len if i < n_segments - 1 else self.length
            segments.append(Signal(self._samples[start:end], self._sample_rate))
        return segments

    def overlap_add_with(self, other, position):
        """Add other signal starting at position."""
        result = list(self._samples)
        while len(result) < position + other.length:
            result.append(0.0)
        for i in range(other.length):
            result[position + i] += other._samples[i]
        return Signal(result, self._sample_rate)

    def window(self, window_func):
        """Apply window function to signal."""
        w = window_func(self.length)
        return Signal([s * wi for s, wi in zip(self._samples, w)], self._sample_rate)

    def moving_rms(self, window_size):
        """Compute moving RMS over the signal."""
        import math
        n = self.length
        result = []
        for i in range(n):
            lo = max(0, i - window_size // 2)
            hi = min(n, i + window_size // 2 + 1)
            frame = self._samples[lo:hi]
            result.append(math.sqrt(sum(s*s for s in frame) / len(frame)))
        return result

    def spectral_centroid(self):
        """Quick spectral centroid computation."""
        from dravex.transform.fft import rfft
        mags = rfft(list(self._samples))
        total = sum(mags)
        if total < 1e-12:
            return 0.0
        weighted = sum(i * m for i, m in enumerate(mags))
        bin_hz = self._sample_rate / self.length
        return (weighted / total) * bin_hz

    def auto_correlate(self, max_lag=None):
        """Autocorrelation of the signal."""
        n = self.length
        if max_lag is None:
            max_lag = n
        result = []
        for lag in range(min(max_lag, n)):
            s = 0.0
            for i in range(n - lag):
                s += self._samples[i] * self._samples[i + lag]
            result.append(s)
        return result

    def resample(self, target_rate):
        """Resample signal to target sample rate."""
        from dravex.resample.interpolate import resample_ratio
        ratio = target_rate / self._sample_rate
        new_samples = resample_ratio(list(self._samples), ratio)
        return Signal(new_samples, target_rate)

    def filter_lowpass(self, cutoff, order=32):
        """Apply lowpass filter to signal."""
        from dravex.filter.fir import fir_lowpass, apply_fir
        h = fir_lowpass(order, cutoff, self._sample_rate)
        filtered = apply_fir(list(self._samples), h)
        return Signal(filtered, self._sample_rate)

    def filter_highpass(self, cutoff, order=32):
        """Apply highpass filter to signal."""
        from dravex.filter.fir import fir_highpass, apply_fir
        h = fir_highpass(order, cutoff, self._sample_rate)
        filtered = apply_fir(list(self._samples), h)
        return Signal(filtered, self._sample_rate)

    def spectrogram(self, frame_size=1024, hop_size=256):
        """Compute spectrogram of signal."""
        from dravex.transform.stft import spectrogram as stft_spec
        return stft_spec(list(self._samples), frame_size, hop_size)

    def mfcc(self, n_coeffs=13, n_fft=512):
        """Compute MFCCs for this signal."""
        from dravex.spectral.mfcc import mfcc as compute_mfcc
        return compute_mfcc(list(self._samples), self._sample_rate, n_coeffs, n_fft=n_fft)

    def pitch(self):
        """Detect fundamental frequency."""
        from dravex.spectral.pitch import autocorrelation_pitch
        return autocorrelation_pitch(list(self._samples), self._sample_rate)

    def detect_onsets(self, frame_size=1024, hop_size=512):
        """Detect onset events in signal."""
        from dravex.detect.onset import energy_onset, pick_onsets
        odf = energy_onset(list(self._samples), frame_size, hop_size)
        return pick_onsets(odf)

    def to_frames(self, frame_size=1024, hop_size=512, window=None):
        """Extract windowed frames."""
        from dravex.core.frame import extract_frames
        return extract_frames(self, frame_size, hop_size, window)

    @staticmethod
    def from_function(func, duration, sample_rate=44100):
        """Generate signal from a function f(t)."""
        n = int(duration * sample_rate)
        samples = [func(i / sample_rate) for i in range(n)]
        return Signal(samples, sample_rate)

    def crossfade_with(self, other, crossfade_samples):
        """Crossfade into another signal."""
        result = list(self._samples)
        other_s = other.samples
        cf = min(crossfade_samples, self.length, other.length)
        for i in range(cf):
            t = i / cf
            idx = self.length - cf + i
            if 0 <= idx < self.length:
                result[idx] = result[idx] * (1 - t) + other_s[i] * t
        result.extend(other_s[cf:])
        return Signal(result, self._sample_rate)

    def repeat(self, n_times):
        """Repeat signal n times."""
        return Signal(list(self._samples) * n_times, self._sample_rate)

    def zero_pad_to_power_of_2(self):
        """Zero-pad to next power of 2 length."""
        from dravex.constants import next_power_of_2
        target = next_power_of_2(self.length)
        return self.pad_zeros(target)

    def dc_offset(self):
        """Return the DC offset (mean value)."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / self.length

    def peak_to_peak(self):
        """Peak-to-peak amplitude."""
        if not self._samples:
            return 0.0
        return max(self._samples) - min(self._samples)

    def clip_count(self, threshold=0.99):
        """Count samples at or near clipping."""
        return sum(1 for s in self._samples if abs(s) >= threshold)

    def histogram(self, n_bins=256):
        """Amplitude histogram of signal."""
        from dravex.stats.histogram import amplitude_histogram
        return amplitude_histogram(list(self._samples), n_bins)

    def entropy(self):
        """Shannon entropy of amplitude distribution."""
        from dravex.stats.entropy import shannon_entropy
        return shannon_entropy(list(self._samples))
