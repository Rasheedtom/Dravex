"""Tukey (cosine-tapered) window function."""
import math
from dravex.constants import PI, TAU

def tukey(n, alpha=0.5):
    """Generate Tukey window. alpha=0 -> rectangular, alpha=1 -> Hann."""
    if n <= 1:
        return [1.0]
    result = []
    for i in range(n):
        if i < alpha * (n - 1) / 2:
            val = 0.5 * (1 + math.cos(PI * (2 * i / (alpha * (n - 1)) - 1)))
        elif i <= (n - 1) * (1 - alpha / 2):
            val = 1.0
        else:
            val = 0.5 * (1 + math.cos(PI * (2 * i / (alpha * (n - 1)) - 2 / alpha + 1)))
        result.append(val)
    return result

def planck_taper(n, epsilon=0.1):
    """Planck-taper window (smooth transition at edges)."""
    if n <= 1:
        return [1.0]
    result = [0.0] * n
    for i in range(n):
        t = i / (n - 1)
        if t <= 0 or t >= 1:
            result[i] = 0.0
        elif t < epsilon:
            z = epsilon * (1.0 / t + 1.0 / (t - epsilon))
            result[i] = 1.0 / (math.exp(z) + 1)
        elif t > 1 - epsilon:
            z = epsilon * (1.0 / (1 - t) + 1.0 / (1 - t - epsilon))
            result[i] = 1.0 / (math.exp(z) + 1)
        else:
            result[i] = 1.0
    return result

def apply_tukey(x, alpha=0.5):
    """Apply Tukey window to signal."""
    w = tukey(len(x), alpha)
    return [s * wi for s, wi in zip(x, w)]

def cosine_window(n):
    """Cosine (sine) window: w[i] = sin(pi*i/(N-1))."""
    if n <= 1:
        return [1.0]
    return [math.sin(PI * i / (n - 1)) for i in range(n)]

def welch_window(n):
    """Welch (parabolic) window."""
    if n <= 1:
        return [1.0]
    M = (n - 1) / 2
    return [1.0 - ((i - M) / M) ** 2 for i in range(n)]
