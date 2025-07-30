"""Periodicity detection in signals."""
import math
from dravex.constants import DEFAULT_EPSILON

def autocorrelation_periodicity(x, min_period=2, max_period=None):
    """Detect dominant period via autocorrelation."""
    n = len(x)
    if max_period is None:
        max_period = n // 2
    best_corr = 0.0
    best_period = 0
    m = sum(x) / n
    centered = [s - m for s in x]
    energy = sum(s * s for s in centered)
    if energy < DEFAULT_EPSILON:
        return 0
    for lag in range(min_period, min(max_period, n // 2)):
        corr = sum(centered[i] * centered[i + lag] for i in range(n - lag))
        normalized = corr / energy
        if normalized > best_corr:
            best_corr = normalized
            best_period = lag
    return best_period if best_corr > 0.3 else 0

def spectral_periodicity(x, sample_rate=44100):
    """Detect periodicity from spectral peaks."""
    from dravex.transform.fft import rfft
    mags = rfft(x)
    if len(mags) < 3:
        return 0.0
    mags[0] = 0
    peak_idx = max(range(1, len(mags)), key=lambda i: mags[i])
    if mags[peak_idx] < DEFAULT_EPSILON:
        return 0.0
    freq = peak_idx * sample_rate / (2 * (len(mags) - 1))
    return sample_rate / freq if freq > DEFAULT_EPSILON else 0.0

def period_strength(x, period):
    """Measure how periodic signal is at given period."""
    n = len(x)
    if period < 1 or period >= n // 2:
        return 0.0
    m = sum(x) / n
    centered = [s - m for s in x]
    energy = sum(s * s for s in centered)
    if energy < DEFAULT_EPSILON:
        return 0.0
    corr = sum(centered[i] * centered[i + period] for i in range(n - period))
    return corr / energy

def harmonic_product_spectrum(x, n_harmonics=5):
    """HPS for robust periodicity detection."""
    from dravex.transform.fft import rfft
    mags = rfft(x)
    n = len(mags)
    hps = list(mags)
    for h in range(2, n_harmonics + 1):
        for i in range(n // h):
            hps[i] *= mags[i * h] if i * h < n else 0
    if not hps:
        return 0
    return max(range(1, len(hps) // 2), key=lambda i: hps[i])

def dominant_period_fft(x, sample_rate=44100):
    """Find dominant period via FFT peak detection."""
    from dravex.transform.fft import rfft
    mags = rfft(x)
    if len(mags) < 3:
        return 0.0
    mags[0] = 0
    peak_idx = max(range(1, len(mags)), key=lambda i: mags[i])
    if mags[peak_idx] < 1e-12:
        return 0.0
    freq = peak_idx * sample_rate / (2 * (len(mags) - 1))
    return sample_rate / freq if freq > 1e-12 else 0.0

def is_periodic(x, threshold=0.5, min_period=2, max_period=None):
    """Check if signal is periodic above threshold."""
    period = autocorrelation_periodicity(x, min_period, max_period)
    if period <= 0:
        return False
    strength = period_strength(x, period)
    return strength > threshold

def period_jitter(x, expected_period, tolerance=0.2):
    """Measure period jitter (variation from expected period)."""
    n = len(x)
    crossings = []
    for i in range(1, n):
        if x[i-1] < 0 and x[i] >= 0:
            crossings.append(i)
    if len(crossings) < 3:
        return 0.0
    periods = [crossings[i+1] - crossings[i] for i in range(len(crossings)-1)]
    if not periods:
        return 0.0
    mean_p = sum(periods) / len(periods)
    import math
    jitter = math.sqrt(sum((p - mean_p)**2 for p in periods) / len(periods))
    return jitter / mean_p
