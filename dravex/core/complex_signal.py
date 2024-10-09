"""Complex-valued signal support."""
import math
from dravex.constants import DEFAULT_EPSILON

class ComplexSignal:
    """Complex-valued discrete signal for analytic signal processing."""

    def __init__(self, real, imag=None, sample_rate=44100):
        self._real = list(real)
        self._imag = list(imag) if imag else [0.0] * len(self._real)
        self._sample_rate = sample_rate

    @property
    def real(self):
        return list(self._real)

    @property
    def imag(self):
        return list(self._imag)

    @property
    def length(self):
        return len(self._real)

    @property
    def sample_rate(self):
        return self._sample_rate

    def magnitude(self):
        return [math.sqrt(r*r + i*i) for r, i in zip(self._real, self._imag)]

    def phase(self):
        return [math.atan2(i, r) for r, i in zip(self._real, self._imag)]

    def conjugate(self):
        return ComplexSignal(self._real, [-i for i in self._imag], self._sample_rate)

    def __add__(self, other):
        return ComplexSignal(
            [a + b for a, b in zip(self._real, other._real)],
            [a + b for a, b in zip(self._imag, other._imag)],
            self._sample_rate)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ComplexSignal(
                [r * other for r in self._real],
                [i * other for i in self._imag],
                self._sample_rate)
        result_real = []
        result_imag = []
        for r1, i1, r2, i2 in zip(self._real, self._imag, other._real, other._imag):
            result_real.append(r1*r2 - i1*i2)
            result_imag.append(r1*i2 + i1*r2)
        return ComplexSignal(result_real, result_imag, self._sample_rate)

    def energy(self):
        return sum(r*r + i*i for r, i in zip(self._real, self._imag))

    def rms(self):
        if not self._real:
            return 0.0
        return math.sqrt(self.energy() / self.length)

    @staticmethod
    def from_polar(magnitudes, phases, sample_rate=44100):
        real = [m * math.cos(p) for m, p in zip(magnitudes, phases)]
        imag = [m * math.sin(p) for m, p in zip(magnitudes, phases)]
        return ComplexSignal(real, imag, sample_rate)

    @staticmethod
    def from_analytic(real_signal, sample_rate=44100):
        from dravex.transform.hilbert import hilbert_transform
        ht = hilbert_transform(real_signal)
        return ComplexSignal(real_signal, ht, sample_rate)

    def fft(self):
        """Compute FFT of complex signal."""
        from dravex.transform.fft import fft
        n = self.length
        X = fft(self._real)
        return [(re, im) for re, im in X]

    def power_spectrum(self):
        """Power spectrum of complex signal."""
        return [r*r + i*i for r, i in zip(self._real, self._imag)]

    def normalize(self):
        """Normalize to unit energy."""
        import math
        e = self.energy()
        if e < 1e-12:
            return self
        scale = 1.0 / math.sqrt(e)
        return self * scale

    def unwrap_phase(self):
        """Unwrap phase of complex signal."""
        import math
        phases = self.phase()
        unwrapped = [phases[0]]
        for i in range(1, len(phases)):
            diff = phases[i] - phases[i-1]
            while diff > math.pi: diff -= 2*math.pi
            while diff < -math.pi: diff += 2*math.pi
            unwrapped.append(unwrapped[-1] + diff)
        return unwrapped

    def rotate(self, angle):
        """Rotate complex signal by angle radians."""
        import math
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_re = [r*cos_a - i*sin_a for r, i in zip(self._real, self._imag)]
        new_im = [r*sin_a + i*cos_a for r, i in zip(self._real, self._imag)]
        return ComplexSignal(new_re, new_im, self._sample_rate)

    def downsample(self, factor):
        """Downsample complex signal."""
        return ComplexSignal(self._real[::factor], self._imag[::factor], self._sample_rate // factor)
