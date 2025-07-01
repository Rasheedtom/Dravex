"""Zero-crossing rate analysis."""

def zero_crossing_rate(x, frame_size=None):
    """Compute zero-crossing rate."""
    if frame_size is None:
        count = 0
        for i in range(1, len(x)):
            if x[i-1] * x[i] < 0:
                count += 1
        return count / max(1, len(x) - 1)
    n = len(x)
    rates = []
    pos = 0
    while pos + frame_size <= n:
        count = 0
        for i in range(pos + 1, pos + frame_size):
            if x[i-1] * x[i] < 0:
                count += 1
        rates.append(count / (frame_size - 1))
        pos += frame_size
    return rates

def zero_crossings(x):
    """Find indices where signal crosses zero."""
    crossings = []
    for i in range(1, len(x)):
        if x[i-1] * x[i] < 0:
            t = abs(x[i-1]) / (abs(x[i-1]) + abs(x[i]))
            crossings.append(i - 1 + t)
    return crossings

def average_zero_crossing_interval(x):
    """Average distance between zero crossings."""
    zc = zero_crossings(x)
    if len(zc) < 2:
        return 0.0
    intervals = [zc[i+1] - zc[i] for i in range(len(zc)-1)]
    return sum(intervals) / len(intervals)

def classify_voiced_unvoiced(x, frame_size=256, zcr_threshold=0.1, energy_threshold=0.01):
    """Classify frames as voiced or unvoiced based on ZCR and energy."""
    n = len(x)
    labels = []
    pos = 0
    while pos + frame_size <= n:
        frame = x[pos:pos + frame_size]
        zcr = sum(1 for i in range(1, len(frame)) if frame[i-1]*frame[i] < 0) / (frame_size - 1)
        energy = sum(s*s for s in frame) / frame_size
        if zcr < zcr_threshold and energy > energy_threshold:
            labels.append("voiced")
        else:
            labels.append("unvoiced")
        pos += frame_size
    return labels
