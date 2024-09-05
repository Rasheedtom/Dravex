"""Frequency-domain spectrum representation."""
import math
from dravex.constants import DEFAULT_EPSILON, amplitude_to_db

class Spectrum:
    """Magnitude and phase spectrum."""

    def __init__(self, magnitudes, phases=None, sample_rate=44100, n_fft=None):
        self._magnitudes = list(magnitudes)
        self._phases = list(phases) if phases else [0.0] * len(magnitudes)
        self._sample_rate = sample_rate
        self._n_fft = n_fft or (len(magnitudes) - 1) * 2

    @property
    def magnitudes(self):
        return list(self._magnitudes)

    @property
    def phases(self):
        return list(self._phases)

    @property
    def n_bins(self):
        return len(self._magnitudes)

    @property
    def frequency_resolution(self):
        return self._sample_rate / self._n_fft

    def bin_to_hz(self, bin_idx):
        return bin_idx * self.frequency_resolution

    def hz_to_bin(self, hz):
        return int(round(hz / self.frequency_resolution))

    def magnitude_db(self):
        return [amplitude_to_db(m) for m in self._magnitudes]

    def power_spectrum(self):
        return [m * m for m in self._magnitudes]

    def peak_frequency(self):
        if not self._magnitudes:
            return 0.0
        max_idx = max(range(len(self._magnitudes)), key=lambda i: self._magnitudes[i])
        return self.bin_to_hz(max_idx)

    def spectral_centroid(self):
        total = sum(self._magnitudes)
        if total < DEFAULT_EPSILON:
            return 0.0
        weighted = sum(self.bin_to_hz(i) * m for i, m in enumerate(self._magnitudes))
        return weighted / total

    def spectral_rolloff(self, threshold=0.85):
        total = sum(self._magnitudes)
        if total < DEFAULT_EPSILON:
            return 0.0
        cumulative = 0.0
        for i, m in enumerate(self._magnitudes):
            cumulative += m
            if cumulative >= threshold * total:
                return self.bin_to_hz(i)
        return self.bin_to_hz(self.n_bins - 1)

    def spectral_flatness(self):
        n = len(self._magnitudes)
        if n == 0:
            return 0.0
        powers = [m * m for m in self._magnitudes if m > DEFAULT_EPSILON]
        if not powers:
            return 0.0
        log_mean = sum(math.log(p) for p in powers) / len(powers)
        arith_mean = sum(powers) / len(powers)
        if arith_mean < DEFAULT_EPSILON:
            return 0.0
        return math.exp(log_mean) / arith_mean

    def band_energy(self, low_hz, high_hz):
        """Compute energy within a frequency band."""
        lo_bin = max(0, self.hz_to_bin(low_hz))
        hi_bin = min(self.n_bins, self.hz_to_bin(high_hz))
        return sum(m * m for m in self._magnitudes[lo_bin:hi_bin])

    def to_complex(self):
        return [m * (math.cos(p) + 1j * math.sin(p))
                for m, p in zip(self._magnitudes, self._phases)]

    def smooth(self, window_size=5):
        """Smooth magnitude spectrum with moving average."""
        n = self.n_bins
        smoothed = []
        for i in range(n):
            lo = max(0, i - window_size // 2)
            hi = min(n, i + window_size // 2 + 1)
            avg = sum(self._magnitudes[lo:hi]) / (hi - lo)
            smoothed.append(avg)
        return Spectrum(smoothed, self._phases, self._sample_rate, self._n_fft)

    def normalize(self):
        """Normalize magnitudes to sum to 1."""
        total = sum(self._magnitudes)
        if total < DEFAULT_EPSILON:
            return self
        normalized = [m / total for m in self._magnitudes]
        return Spectrum(normalized, self._phases, self._sample_rate, self._n_fft)

    def top_k_bins(self, k=5):
        """Return indices of k strongest frequency bins."""
        indexed = list(enumerate(self._magnitudes))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(idx, self._magnitudes[idx], self.bin_to_hz(idx)) for idx, _ in indexed[:k]]

    def spectral_slope(self):
        """Linear regression slope of log-magnitude spectrum."""
        import math
        n = self.n_bins
        if n < 2:
            return 0.0
        log_mags = [math.log(max(m, DEFAULT_EPSILON)) for m in self._magnitudes]
        x_mean = (n - 1) / 2
        y_mean = sum(log_mags) / n
        num = sum((i - x_mean) * (log_mags[i] - y_mean) for i in range(n))
        den = sum((i - x_mean)**2 for i in range(n))
        return num / den if abs(den) > DEFAULT_EPSILON else 0.0
