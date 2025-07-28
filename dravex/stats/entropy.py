"""Signal entropy measures."""
import math
from dravex.constants import DEFAULT_EPSILON

def shannon_entropy(x, n_bins=256):
    """Shannon entropy of signal amplitude distribution.

    Returns entropy in bits (log base 2).
    """
    """Shannon entropy of signal amplitude distribution."""
    if not x:
        return 0.0
    lo, hi = min(x), max(x)
    if lo == hi:
        return 0.0
    step = (hi - lo) / n_bins
    counts = [0] * n_bins
    for s in x:
        idx = min(int((s - lo) / step), n_bins - 1)
        counts[idx] += 1
    n = len(x)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)
    return entropy

def spectral_entropy(magnitudes):
    """Entropy of normalized power spectrum."""
    total = sum(m * m for m in magnitudes)
    if total < DEFAULT_EPSILON:
        return 0.0
    probs = [(m * m) / total for m in magnitudes]
    entropy = 0.0
    for p in probs:
        if p > DEFAULT_EPSILON:
            entropy -= p * math.log2(p)
    return entropy

def sample_entropy(x, m=2, r=None):
    """Sample entropy (measure of signal complexity)."""
    n = len(x)
    if r is None:
        r = 0.2 * _std(x)
    if r < DEFAULT_EPSILON or n < m + 1:
        return 0.0
    count_m = _count_matches(x, m, r)
    count_m1 = _count_matches(x, m + 1, r)
    if count_m == 0 or count_m1 == 0:
        return 0.0
    return -math.log(count_m1 / count_m)

def _count_matches(x, template_len, tolerance):
    n = len(x)
    count = 0
    for i in range(n - template_len):
        for j in range(i + 1, n - template_len):
            match = True
            for k in range(template_len):
                if abs(x[i+k] - x[j+k]) > tolerance:
                    match = False
                    break
            if match:
                count += 1
    return count

def _std(x):
    if len(x) < 2:
        return 0.0
    m = sum(x) / len(x)
    return math.sqrt(sum((s - m)**2 for s in x) / (len(x) - 1))

def permutation_entropy(x, order=3, delay=1):
    """Permutation entropy (complexity from ordinal patterns)."""
    n = len(x)
    patterns = {}
    for i in range(n - (order - 1) * delay):
        window = tuple(x[i + k * delay] for k in range(order))
        ranked = tuple(sorted(range(order), key=lambda k: window[k]))
        patterns[ranked] = patterns.get(ranked, 0) + 1
    total = sum(patterns.values())
    entropy = 0.0
    for count in patterns.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(math.factorial(order))
    return entropy / max_entropy if max_entropy > 0 else 0.0
