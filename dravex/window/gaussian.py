"""Gaussian window function."""
import math

def gaussian_window(n, sigma=0.4):
    """Generate Gaussian window of length n."""
    if n <= 1:
        return [1.0]
    M = (n - 1) / 2
    result = []
    for i in range(n):
        x = (i - M) / (sigma * M)
        result.append(math.exp(-0.5 * x * x))
    return result

def confined_gaussian(n, sigma=0.4):
    """Confined Gaussian window (forced to zero at boundaries)."""
    w = gaussian_window(n, sigma)
    edge = w[0]
    return [max(0, wi - edge) for wi in w]

def apply_gaussian(x, sigma=0.4):
    """Apply Gaussian window to signal."""
    w = gaussian_window(len(x), sigma)
    return [s * wi for s, wi in zip(x, w)]

def gaussian_derivative_window(n, sigma=0.4, order=1):
    """Derivative of Gaussian window (for edge detection)."""
    if n <= 1:
        return [0.0]
    M = (n - 1) / 2
    result = []
    for i in range(n):
        x = (i - M) / (sigma * M)
        g = math.exp(-0.5 * x * x)
        if order == 1:
            result.append(-x / (sigma * M) * g)
        elif order == 2:
            result.append((x * x / (sigma * M)**2 - 1/(sigma * M)**2) * g)
        else:
            result.append(g)
    return result

def gabor_window(n, sigma=0.4, freq=0.1):
    """Gabor window (Gaussian * complex sinusoid), returns real part."""
    M = (n - 1) / 2
    result = []
    for i in range(n):
        x = (i - M) / (sigma * M) if sigma * M > 0 else 0
        g = math.exp(-0.5 * x * x)
        result.append(g * math.cos(2 * math.pi * freq * i))
    return result
