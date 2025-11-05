"""Statistical moments for signal analysis."""
import math
from dravex.constants import DEFAULT_EPSILON

def mean(x):
    if not x:
        return 0.0
    return sum(x) / len(x)

def variance(x):
    """Sample variance using two-pass algorithm."""
    n = len(x)
    if n < 2:
        return 0.0
    m = mean(x)
    return sum((s - m)**2 for s in x) / (len(x) - 1)

def std_dev(x):
    return math.sqrt(variance(x))

def skewness(x):
    n = len(x)
    if n < 3:
        return 0.0
    m = mean(x)
    s = std_dev(x)
    if s < DEFAULT_EPSILON:
        return 0.0
    return sum(((xi - m) / s)**3 for xi in x) * n / ((n-1) * (n-2))

def kurtosis(x):
    n = len(x)
    if n < 4:
        return 0.0
    m = mean(x)
    s = std_dev(x)
    if s < DEFAULT_EPSILON:
        return 0.0
    k4 = sum(((xi - m) / s)**4 for xi in x) / n
    return k4 - 3.0

def central_moment(x, order):
    m = mean(x)
    return sum((xi - m)**order for xi in x) / max(1, len(x))

def rms(x):
    if not x:
        return 0.0
    return math.sqrt(sum(s*s for s in x) / len(x))

def crest_factor(x):
    r = rms(x)
    if r < DEFAULT_EPSILON:
        return 0.0
    return max(abs(s) for s in x) / r

def median(x):
    """Compute median."""
    s = sorted(x)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def mad(x):
    """Median absolute deviation."""
    med = median(x)
    return median([abs(xi - med) for xi in x])

def trimmed_mean(x, trim_fraction=0.1):
    """Trimmed mean (removes outliers)."""
    s = sorted(x)
    n = len(s)
    trim = int(n * trim_fraction)
    if 2 * trim >= n:
        return mean(s)
    trimmed = s[trim:n - trim]
    return sum(trimmed) / len(trimmed)

def geometric_mean(x):
    """Geometric mean (for positive values)."""
    import math
    n = len(x)
    if n == 0:
        return 0.0
    log_sum = sum(math.log(max(xi, 1e-15)) for xi in x)
    return math.exp(log_sum / n)

def harmonic_mean(x):
    """Harmonic mean (for positive values)."""
    n = len(x)
    if n == 0:
        return 0.0
    recip_sum = sum(1.0 / max(xi, 1e-15) for xi in x)
    return n / recip_sum

def peak_to_average_ratio(x):
    """Peak-to-average power ratio (PAPR)."""
    import math
    if not x:
        return 0.0
    peak = max(abs(s) for s in x)
    avg_power = sum(s * s for s in x) / len(x)
    if avg_power < 1e-15:
        return 0.0
    return peak * peak / avg_power

def zero_crossing_rate(x):
    """Zero crossing rate."""
    if len(x) < 2:
        return 0.0
    count = sum(1 for i in range(1, len(x)) if x[i-1] * x[i] < 0)
    return count / (len(x) - 1)
