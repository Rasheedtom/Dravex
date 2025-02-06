"""Spectral bandwidth and rolloff."""
import math
from dravex.transform.fft import rfft
from dravex.constants import DEFAULT_EPSILON

def spectral_bandwidth(magnitudes, freqs, centroid=None):
    """Spectral bandwidth (weighted std dev of frequencies)."""
    total = sum(magnitudes)
    if total < DEFAULT_EPSILON:
        return 0.0
    if centroid is None:
        centroid = sum(f * m for f, m in zip(freqs, magnitudes)) / total
    variance = sum(m * (f - centroid)**2 for f, m in zip(freqs, magnitudes)) / total
    return math.sqrt(variance)

def spectral_rolloff(magnitudes, freqs, threshold=0.85):
    """Frequency below which threshold% of spectral energy lies."""
    total = sum(m * m for m in magnitudes)
    if total < DEFAULT_EPSILON:
        return 0.0
    cumulative = 0.0
    for i, m in enumerate(magnitudes):
        cumulative += m * m
        if cumulative >= threshold * total:
            return freqs[i] if i < len(freqs) else 0.0
    return freqs[-1] if freqs else 0.0

def spectral_centroid(magnitudes, freqs):
    """First spectral moment (center of mass)."""
    total = sum(magnitudes)
    if total < DEFAULT_EPSILON:
        return 0.0
    return sum(f * m for f, m in zip(freqs, magnitudes)) / total

def spectral_spread(magnitudes, freqs):
    """Same as bandwidth, alias."""
    return spectral_bandwidth(magnitudes, freqs)

def spectral_skewness(magnitudes, freqs):
    """Third standardized moment of spectrum."""
    total = sum(magnitudes)
    if total < DEFAULT_EPSILON:
        return 0.0
    mu = sum(f * m for f, m in zip(freqs, magnitudes)) / total
    var = sum(m * (f - mu)**2 for f, m in zip(freqs, magnitudes)) / total
    if var < DEFAULT_EPSILON:
        return 0.0
    sigma = math.sqrt(var)
    return sum(m * ((f - mu) / sigma)**3 for f, m in zip(freqs, magnitudes)) / total

def spectral_kurtosis(magnitudes, freqs):
    """Fourth standardized moment of spectrum."""
    total = sum(magnitudes)
    if total < DEFAULT_EPSILON:
        return 0.0
    mu = sum(f * m for f, m in zip(freqs, magnitudes)) / total
    var = sum(m * (f - mu)**2 for f, m in zip(freqs, magnitudes)) / total
    if var < DEFAULT_EPSILON:
        return 0.0
    sigma = math.sqrt(var)
    return sum(m * ((f - mu) / sigma)**4 for f, m in zip(freqs, magnitudes)) / total

def spectral_crest(magnitudes):
    """Spectral crest factor: peak / mean."""
    if not magnitudes:
        return 0.0
    peak = max(magnitudes)
    mean = sum(magnitudes) / len(magnitudes)
    if mean < 1e-12:
        return 0.0
    return peak / mean

def spectral_decrease(magnitudes, freqs):
    """Spectral decrease (perceptual tilt measure)."""
    if len(magnitudes) < 2:
        return 0.0
    numerator = sum((magnitudes[k] - magnitudes[0]) / k for k in range(1, len(magnitudes)))
    denominator = sum(magnitudes[1:])
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator

def spectral_entropy(magnitudes):
    """Entropy of normalized spectral distribution."""
    import math
    total = sum(m * m for m in magnitudes)
    if total < 1e-12:
        return 0.0
    probs = [(m * m) / total for m in magnitudes]
    return -sum(p * math.log2(p) for p in probs if p > 1e-12)

def spectral_irregularity(magnitudes):
    """Jensen spectral irregularity."""
    n = len(magnitudes)
    if n < 3:
        return 0.0
    diff_sum = 0.0
    mag_sum = sum(m * m for m in magnitudes)
    for k in range(1, n - 1):
        local_avg = (magnitudes[k-1] + magnitudes[k] + magnitudes[k+1]) / 3
        diff_sum += (magnitudes[k] - local_avg) ** 2
    if mag_sum < 1e-12:
        return 0.0
    return diff_sum / mag_sum

def spectral_flux(magnitudes_prev, magnitudes_curr):
    """Spectral flux between consecutive frames."""
    import math
    flux = 0.0
    for prev, curr in zip(magnitudes_prev, magnitudes_curr):
        diff = curr - prev
        if diff > 0:
            flux += diff * diff
    return math.sqrt(flux)

def spectral_contrast_ratio(magnitudes, n_bands=6, alpha=0.02):
    """Spectral contrast: ratio of peaks to valleys in each sub-band."""
    import math
    n = len(magnitudes)
    band_size = n // n_bands
    contrasts = []
    for b in range(n_bands):
        start = b * band_size
        end = min(start + band_size, n)
        band = sorted(magnitudes[start:end])
        if not band:
            contrasts.append(0.0)
            continue
        n_alpha = max(1, int(len(band) * alpha))
        peak = sum(band[-n_alpha:]) / n_alpha
        valley = sum(band[:n_alpha]) / n_alpha
        if valley < 1e-12:
            contrasts.append(math.log10(max(peak, 1e-12)))
        else:
            contrasts.append(math.log10(peak / valley))
    return contrasts

def spectral_tilt(magnitudes, freqs):
    """Spectral tilt via linear regression of log-magnitude."""
    import math
    n = len(magnitudes)
    if n < 2:
        return 0.0
    log_mags = [math.log(max(m, 1e-12)) for m in magnitudes]
    log_freqs = [math.log(max(f, 1)) for f in freqs if f > 0]
    n_valid = min(len(log_mags), len(log_freqs))
    if n_valid < 2:
        return 0.0
    x = log_freqs[:n_valid]
    y = log_mags[:n_valid]
    x_mean = sum(x) / n_valid
    y_mean = sum(y) / n_valid
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean)**2 for xi in x)
    return num / den if abs(den) > 1e-12 else 0.0

def spectral_moments(magnitudes, freqs, order=4):
    """Compute spectral moments up to given order."""
    total = sum(magnitudes)
    if total < 1e-12:
        return [0.0] * order
    centroid = sum(f * m for f, m in zip(freqs, magnitudes)) / total
    moments = [centroid]
    for p in range(2, order + 1):
        moment = sum(m * (f - centroid)**p for f, m in zip(freqs, magnitudes)) / total
        moments.append(moment)
    return moments
