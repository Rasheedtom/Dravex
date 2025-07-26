"""Amplitude histogram and distribution analysis."""

def amplitude_histogram(x, n_bins=256):
    """Compute amplitude histogram."""
    if not x:
        return [], []
    lo, hi = min(x), max(x)
    if lo == hi:
        return [lo], [len(x)]
    step = (hi - lo) / n_bins
    bins = [lo + i * step for i in range(n_bins)]
    counts = [0] * n_bins
    for s in x:
        idx = min(int((s - lo) / step), n_bins - 1)
        counts[idx] += 1
    return bins, counts

def cumulative_distribution(x, n_bins=256):
    """Compute cumulative distribution function."""
    bins, counts = amplitude_histogram(x, n_bins)
    total = sum(counts)
    if total == 0:
        return bins, [0.0] * len(bins)
    cumulative = []
    running = 0
    for c in counts:
        running += c
        cumulative.append(running / total)
    return bins, cumulative

def percentile(x, p):
    """Compute p-th percentile."""
    if not x:
        return 0.0
    sorted_x = sorted(x)
    idx = p / 100 * (len(sorted_x) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_x) - 1)
    frac = idx - lo
    return sorted_x[lo] * (1 - frac) + sorted_x[hi] * frac

def interquartile_range(x):
    """Compute IQR (Q3 - Q1)."""
    return percentile(x, 75) - percentile(x, 25)

def dynamic_range(x):
    """Dynamic range in dB."""
    import math
    from dravex.constants import amplitude_to_db
    if not x:
        return 0.0
    peak = max(abs(s) for s in x)
    rms_val = math.sqrt(sum(s*s for s in x) / len(x))
    if rms_val < 1e-15:
        return 0.0
    return amplitude_to_db(peak) - amplitude_to_db(rms_val)
