"""Audio clipping detection."""
from dravex.constants import DEFAULT_EPSILON

def detect_clipping(x, threshold=0.99):
    """Detect samples at or near clipping level."""
    clipped = []
    for i, s in enumerate(x):
        if abs(s) >= threshold:
            clipped.append(i)
    return clipped

def clipping_ratio(x, threshold=0.99):
    """Fraction of samples that are clipped."""
    if not x:
        return 0.0
    count = sum(1 for s in x if abs(s) >= threshold)
    return count / len(x)

def consecutive_clipping(x, threshold=0.99, min_run=3):
    """Find consecutive clipping regions."""
    regions = []
    start = None
    for i, s in enumerate(x):
        if abs(s) >= threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_run:
                regions.append((start, i))
            start = None
    if start is not None and len(x) - start >= min_run:
        regions.append((start, len(x)))
    return regions

def headroom_db(x):
    """Headroom in dB (distance from clipping)."""
    import math
    peak = max(abs(s) for s in x) if x else 0
    if peak < DEFAULT_EPSILON:
        return 100.0
    return -20 * math.log10(peak)

def declip(x, threshold=0.99, iterations=10):
    """Simple declipping via interpolation."""
    result = list(x)
    for _ in range(iterations):
        for i in range(1, len(result) - 1):
            if abs(result[i]) >= threshold:
                result[i] = (result[i-1] + result[i+1]) / 2
    return result
