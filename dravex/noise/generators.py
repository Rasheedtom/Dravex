"""Noise signal generators."""
import math
import random as _random

def white_noise(n, amplitude=1.0, rng=None):
    """Generate white (uniform) noise.

    Args:
        n: Number of samples to generate.
        amplitude: Peak amplitude of output signal.
        rng: Optional random.Random instance for reproducibility.
    """
    """Generate white (uniform) noise."""
    if rng is None:
        rng = _random.Random(42)
    return [amplitude * (2 * rng.random() - 1) for _ in range(n)]

def gaussian_noise(n, mean=0.0, std=1.0, rng=None):
    """Generate Gaussian (normal) noise."""
    if rng is None:
        rng = _random.Random(42)
    return [rng.gauss(mean, std) for _ in range(n)]

def pink_noise(n, rng=None):
    """Generate pink (1/f) noise using Voss-McCartney algorithm."""
    if rng is None:
        rng = _random.Random(42)
    n_sources = 8
    sources = [0.0] * n_sources
    result = []
    counter = 0
    for _ in range(n):
        counter += 1
        for j in range(n_sources):
            if counter % (1 << j) == 0:
                sources[j] = rng.random() * 2 - 1
        result.append(sum(sources) / n_sources)
    return result

def brown_noise(n, rng=None):
    """Generate brown (Brownian / red) noise."""
    if rng is None:
        rng = _random.Random(42)
    result = [0.0]
    for _ in range(n - 1):
        step = rng.gauss(0, 0.1)
        val = result[-1] + step
        val = max(-1, min(1, val))
        result.append(val)
    return result

def blue_noise(n, rng=None):
    """Generate blue noise (differentiated white noise)."""
    w = white_noise(n + 1, rng=rng)
    return [w[i + 1] - w[i] for i in range(n)]

def violet_noise(n, rng=None):
    """Generate violet noise (differentiated blue noise)."""
    b = blue_noise(n + 1, rng=rng)
    return [b[i + 1] - b[i] for i in range(min(n, len(b) - 1))]

def impulse_noise(n, density=0.01, amplitude=1.0, rng=None):
    """Generate sparse impulse noise."""
    import random as _r
    if rng is None:
        rng = _r.Random(42)
    result = [0.0] * n
    for i in range(n):
        if rng.random() < density:
            result[i] = amplitude * (1 if rng.random() > 0.5 else -1)
    return result

def chirp_noise(n, min_freq=20, max_freq=20000, sample_rate=44100, rng=None):
    """Generate noise-modulated chirp."""
    import math, random as _r
    if rng is None:
        rng = _r.Random(42)
    result = []
    phase = 0
    for i in range(n):
        t = i / sample_rate
        freq = min_freq + (max_freq - min_freq) * (i / n)
        freq *= (1 + rng.gauss(0, 0.05))
        phase += 2 * math.pi * freq / sample_rate
        result.append(math.sin(phase) * rng.gauss(1, 0.3))
    return result
