"""Colored noise generation with specific spectral shapes."""
import math
from dravex.transform.fft import fft, ifft
from dravex.constants import DEFAULT_EPSILON
import random as _random

def colored_noise(n, exponent=-1.0, rng=None):
    """Generate noise with power spectrum proportional to f^exponent."""
    if rng is None:
        rng = _random.Random(42)
    N = n if n % 2 == 0 else n + 1
    freqs = []
    for k in range(N):
        re = rng.gauss(0, 1)
        im = rng.gauss(0, 1)
        freqs.append((re, im))
    for k in range(1, N // 2):
        f = k
        scale = f ** (exponent / 2.0)
        re, im = freqs[k]
        freqs[k] = (re * scale, im * scale)
        freqs[N - k] = (re * scale, -im * scale)
    freqs[0] = (0.0, 0.0)
    result = ifft(freqs)
    samples = [r for r, _ in result[:n]]
    peak = max(abs(s) for s in samples) or 1.0
    return [s / peak for s in samples]

def bandlimited_noise(n, low_freq, high_freq, sample_rate, rng=None):
    """Generate noise limited to a frequency band."""
    if rng is None:
        rng = _random.Random(42)
    from dravex.noise.generators import white_noise
    from dravex.filter.fir import fir_bandpass, apply_fir
    w = white_noise(n, rng=rng)
    h = fir_bandpass(64, low_freq, high_freq, sample_rate)
    return apply_fir(w, h)

def harmonic_noise(n, fundamental, n_harmonics, sample_rate, jitter=0.02, rng=None):
    """Generate quasi-periodic noise with harmonic structure."""
    if rng is None:
        rng = _random.Random(42)
    result = [0.0] * n
    for h in range(1, n_harmonics + 1):
        freq = fundamental * h * (1 + rng.gauss(0, jitter))
        amp = 1.0 / h
        phase = rng.random() * 2 * math.pi
        for i in range(n):
            t = i / sample_rate
            result[i] += amp * math.sin(2 * math.pi * freq * t + phase)
    peak = max(abs(s) for s in result) or 1.0
    return [s / peak for s in result]

def perlin_noise_1d(n, scale=10, rng=None):
    """1D Perlin-like smooth noise."""
    import math, random as _r
    if rng is None:
        rng = _r.Random(42)
    n_nodes = max(2, n // scale + 2)
    gradients = [rng.random() * 2 - 1 for _ in range(n_nodes)]
    result = []
    for i in range(n):
        x = i / scale
        x0 = int(x) % n_nodes
        x1 = (x0 + 1) % n_nodes
        t = x - int(x)
        t = t * t * (3 - 2 * t)
        v0 = gradients[x0] * (x - int(x))
        v1 = gradients[x1] * (x - int(x) - 1)
        result.append(v0 * (1 - t) + v1 * t)
    return result

def spectral_shaping(white, target_spectrum):
    """Shape white noise to match target spectral envelope."""
    from dravex.transform.fft import fft, ifft
    from dravex.constants import next_power_of_2
    n = len(white)
    N = next_power_of_2(n)
    W = fft(list(white) + [0.0] * (N - n))
    n_bins = N // 2 + 1
    shaped = []
    for k in range(N):
        idx = min(k, N - k) if k > N // 2 else k
        scale = target_spectrum[idx] if idx < len(target_spectrum) else 1.0
        wr, wi = W[k]
        shaped.append((wr * scale, wi * scale))
    result = ifft(shaped)
    return [r for r, _ in result[:n]]
