"""Median filter for impulse noise removal."""

def median_filter(x, kernel_size=3):
    """Apply 1D median filter."""
    n = len(x)
    half = kernel_size // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = sorted(x[lo:hi])
        result.append(window[len(window) // 2])
    return result

def weighted_median_filter(x, kernel_size=5, weights=None):
    """Weighted median filter."""
    n = len(x)
    half = kernel_size // 2
    if weights is None:
        center = half
        weights = [1.0 / (1 + abs(i - center)) for i in range(kernel_size)]
    result = []
    for i in range(n):
        pairs = []
        for j in range(kernel_size):
            idx = i - half + j
            if 0 <= idx < n:
                pairs.append((x[idx], weights[j]))
        pairs.sort(key=lambda p: p[0])
        total_w = sum(w for _, w in pairs)
        cumulative = 0.0
        median_val = pairs[0][0]
        for val, w in pairs:
            cumulative += w
            if cumulative >= total_w / 2:
                median_val = val
                break
        result.append(median_val)
    return result

def adaptive_median(x, min_size=3, max_size=11):
    """Adaptive median filter that grows window to exclude impulses."""
    """Adaptive median filter that grows window until impulse is excluded."""
    n = len(x)
    result = []
    for i in range(n):
        size = min_size
        while size <= max_size:
            half = size // 2
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = sorted(x[lo:hi])
            z_med = window[len(window) // 2]
            z_min = window[0]
            z_max = window[-1]
            if z_min < z_med < z_max:
                if z_min < x[i] < z_max:
                    result.append(x[i])
                else:
                    result.append(z_med)
                break
            size += 2
        else:
            half = max_size // 2
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = sorted(x[lo:hi])
            result.append(window[len(window) // 2])
    return result
