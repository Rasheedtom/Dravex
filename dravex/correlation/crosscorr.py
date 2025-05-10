"""Cross-correlation computation."""
import math
from dravex.constants import DEFAULT_EPSILON

def cross_correlation(x, y, max_lag=None):
    """Compute cross-correlation between x and y."""
    n = min(len(x), len(y))
    if max_lag is None:
        max_lag = n
    result = []
    for lag in range(-max_lag + 1, max_lag):
        s = 0.0
        for i in range(n):
            j = i + lag
            if 0 <= j < len(y):
                s += x[i] * y[j]
        result.append(s)
    return result

def normalized_cross_correlation(x, y, max_lag=None):
    """Normalized cross-correlation bounded to [-1, 1]."""
    """Normalized cross-correlation (Pearson-like)."""
    r = cross_correlation(x, y, max_lag)
    ex = sum(s * s for s in x)
    ey = sum(s * s for s in y)
    norm = math.sqrt(ex * ey)
    if norm < DEFAULT_EPSILON:
        return r
    return [ri / norm for ri in r]

def cross_correlation_fft(x, y):
    """Fast cross-correlation via FFT."""
    from dravex.transform.fft import fft, ifft
    from dravex.constants import next_power_of_2
    n = len(x) + len(y) - 1
    N = next_power_of_2(n)
    X = fft(list(x) + [0.0] * (N - len(x)))
    Y = fft(list(y) + [0.0] * (N - len(y)))
    product = [(xr*yr + xi*yi, xi*yr - xr*yi) for (xr, xi), (yr, yi) in zip(X, Y)]
    r = ifft(product)
    return [r[i][0] for i in range(n)]

def similarity(x, y):
    """Cosine similarity between two signals."""
    dot = sum(a * b for a, b in zip(x, y))
    nx = math.sqrt(sum(a * a for a in x))
    ny = math.sqrt(sum(b * b for b in y))
    if nx < DEFAULT_EPSILON or ny < DEFAULT_EPSILON:
        return 0.0
    return dot / (nx * ny)

def lagged_correlation(x, y, lag):
    """Correlation at a specific lag."""
    n = min(len(x), len(y))
    if lag >= 0:
        s = sum(x[i] * y[i + lag] for i in range(n - lag))
    else:
        s = sum(x[i - lag] * y[i] for i in range(n + lag))
    return s

def pearson_correlation(x, y):
    """Pearson correlation coefficient."""
    import math
    n = min(len(x), len(y))
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    sx = math.sqrt(sum((x[i] - mx)**2 for i in range(n)) / n)
    sy = math.sqrt(sum((y[i] - my)**2 for i in range(n)) / n)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return cov / (sx * sy)

def sliding_correlation(x, y, window_size):
    """Sliding window correlation."""
    n = min(len(x), len(y))
    result = []
    for i in range(n - window_size + 1):
        wx = x[i:i + window_size]
        wy = y[i:i + window_size]
        result.append(pearson_correlation(wx, wy))
    return result
