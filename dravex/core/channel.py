"""Multi-channel signal container."""
from dravex.core.signal import Signal
from dravex.errors import DimensionError

class MultiChannel:
    """Multi-channel signal (e.g. stereo, surround)."""

    def __init__(self, channels, sample_rate=44100):
        self._channels = []
        for ch in channels:
            if isinstance(ch, Signal):
                self._channels.append(ch)
            else:
                self._channels.append(Signal(ch, sample_rate))
        self._sample_rate = sample_rate

    @property
    def n_channels(self):
        return len(self._channels)

    @property
    def length(self):
        return self._channels[0].length if self._channels else 0

    @property
    def sample_rate(self):
        return self._sample_rate

    def channel(self, idx):
        return self._channels[idx]

    def to_mono(self, weights=None):
        """Mix down to mono signal using specified channel weights."""
        """Mix down to mono signal."""
        n = self.n_channels
        if n == 0:
            return Signal([], self._sample_rate)
        if weights is None:
            weights = [1.0 / n] * n
        length = self.length
        result = [0.0] * length
        for ch_idx, w in enumerate(weights):
            samples = self._channels[ch_idx].samples
            for i in range(length):
                result[i] += samples[i] * w
        return Signal(result, self._sample_rate)

    def to_stereo(self):
        """Convert to stereo (duplicate mono or mix down)."""
        if self.n_channels == 2:
            return self
        mono = self.to_mono()
        return MultiChannel([mono, mono], self._sample_rate)

    def split(self):
        return list(self._channels)

    def apply_gain(self, gains):
        """Apply per-channel gain."""
        result = []
        for ch, g in zip(self._channels, gains):
            result.append(ch * g)
        return MultiChannel(result, self._sample_rate)

    def pan(self, position):
        """Stereo panning. position in [-1, 1]. -1=left, 1=right."""
        import math
        mono = self.to_mono()
        left_gain = math.cos((position + 1) * math.pi / 4)
        right_gain = math.sin((position + 1) * math.pi / 4)
        return MultiChannel([mono * left_gain, mono * right_gain], self._sample_rate)

    @staticmethod
    def from_interleaved(samples, n_channels, sample_rate=44100):
        channels = [[] for _ in range(n_channels)]
        for i, s in enumerate(samples):
            channels[i % n_channels].append(s)
        return MultiChannel(channels, sample_rate)

    def to_interleaved(self):
        result = []
        for i in range(self.length):
            for ch in self._channels:
                result.append(ch[i])
        return result

    def mid_side(self):
        """Convert stereo to mid-side representation."""
        if self.n_channels != 2:
            return self
        left = self._channels[0].samples
        right = self._channels[1].samples
        mid = [(l + r) / 2 for l, r in zip(left, right)]
        side = [(l - r) / 2 for l, r in zip(left, right)]
        return MultiChannel([mid, side], self._sample_rate)

    def from_mid_side(self):
        """Convert mid-side back to left-right."""
        if self.n_channels != 2:
            return self
        mid = self._channels[0].samples
        side = self._channels[1].samples
        left = [m + s for m, s in zip(mid, side)]
        right = [m - s for m, s in zip(mid, side)]
        return MultiChannel([left, right], self._sample_rate)

    def correlation(self):
        """Correlation between channels (stereo correlation)."""
        if self.n_channels < 2:
            return 1.0
        from dravex.correlation.crosscorr import similarity
        return similarity(self._channels[0].samples, self._channels[1].samples)

    def peak_normalize_all(self, target=1.0):
        """Normalize all channels to same peak."""
        peak = max(ch.max_amplitude() for ch in self._channels)
        if peak < 1e-12:
            return self
        scale = target / peak
        return self.apply_gain([scale] * self.n_channels)

    def trim_to_shortest(self):
        """Trim all channels to shortest channel length."""
        min_len = min(ch.length for ch in self._channels)
        return MultiChannel([ch[:min_len] for ch in self._channels], self._sample_rate)

    def rms_per_channel(self):
        """RMS for each channel."""
        return [ch.rms() for ch in self._channels]
