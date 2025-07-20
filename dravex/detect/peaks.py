"""Peak detection in signals."""

def find_peaks(x, min_height=None, min_distance=1, prominence=None):
    """Find local maxima in signal."""
    n = len(x)
    if n < 3:
        return []
    peaks = []
    for i in range(1, n - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            if min_height is not None and x[i] < min_height:
                continue
            peaks.append(i)
    if min_distance > 1:
        peaks = _filter_by_distance(x, peaks, min_distance)
    if prominence is not None:
        peaks = _filter_by_prominence(x, peaks, prominence)
    return peaks

def _filter_by_distance(x, peaks, min_dist):
    result = []
    for p in sorted(peaks, key=lambda i: x[i], reverse=True):
        if all(abs(p - r) >= min_dist for r in result):
            result.append(p)
    return sorted(result)

def _filter_by_prominence(x, peaks, min_prom):
    result = []
    n = len(x)
    for p in peaks:
        search_left = max(0, p - 50)
        left_min = min(x[search_left:p]) if p > search_left else x[p]
        right_min = min(x[p:min(n, p+50)]) if p < n-1 else x[p]
        prom = x[p] - max(left_min, right_min)
        if prom >= min_prom:
            result.append(p)
    return result

def find_valleys(x, **kwargs):
    """Find local minima."""
    inverted = [-s for s in x]
    return find_peaks(inverted, **kwargs)

def peak_interpolate(x, peak_idx):
    """Parabolic interpolation for sub-sample peak location."""
    if peak_idx <= 0 or peak_idx >= len(x) - 1:
        return float(peak_idx), x[peak_idx]
    a, b, c = x[peak_idx-1], x[peak_idx], x[peak_idx+1]
    denom = a - 2*b + c
    if abs(denom) < 1e-12:
        return float(peak_idx), b
    offset = 0.5 * (a - c) / denom
    value = b - 0.25 * (a - c) * offset
    return peak_idx + offset, value

def prominence(x, peaks):
    """Compute prominence for each peak."""
    n = len(x)
    result = []
    for p in peaks:
        left_min = min(x[max(0, p-50):p]) if p > 0 else x[p]
        right_min = min(x[p:min(n, p+50)]) if p < n-1 else x[p]
        prom = x[p] - max(left_min, right_min)
        result.append(prom)
    return result

def peak_widths(x, peaks, rel_height=0.5):
    """Estimate peak width at relative height."""
    widths = []
    for p in peaks:
        threshold = x[p] * rel_height
        left = p
        while left > 0 and x[left] > threshold:
            left -= 1
        right = p
        while right < len(x) - 1 and x[right] > threshold:
            right += 1
        widths.append(right - left)
    return widths

def adaptive_threshold_peaks(x, window_size=50, multiplier=2.0):
    """Find peaks using adaptive threshold."""
    n = len(x)
    peaks = []
    for i in range(1, n - 1):
        if x[i] <= x[i-1] or x[i] <= x[i+1]:
            continue
        lo = max(0, i - window_size)
        hi = min(n, i + window_size)
        local_mean = sum(x[lo:hi]) / (hi - lo)
        if x[i] > multiplier * local_mean:
            peaks.append(i)
    return peaks
