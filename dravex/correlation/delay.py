"""Time delay estimation between signals."""
import math
from dravex.constants import DEFAULT_EPSILON

def gcc_phat(x, y, sample_rate=44100):
    """Generalized Cross-Correlation with Phase Transform."""
    from dravex.transform.fft import fft, ifft
    from dravex.constants import next_power_of_2
    n = max(len(x), len(y))
    N = next_power_of_2(n)
    X = fft(list(x) + [0.0] * (N - len(x)))
    Y = fft(list(y) + [0.0] * (N - len(y)))
    cross = []
    for (xr, xi), (yr, yi) in zip(X, Y):
        cr = xr * yr + xi * yi
        ci = xi * yr - xr * yi
        mag = math.sqrt(cr * cr + ci * ci)
        if mag > DEFAULT_EPSILON:
            cross.append((cr / mag, ci / mag))
        else:
            cross.append((0.0, 0.0))
    r = ifft(cross)
    corr = [r[i][0] for i in range(N)]
    peak_idx = max(range(N), key=lambda i: corr[i])
    if peak_idx > N // 2:
        peak_idx -= N
    delay_seconds = peak_idx / sample_rate
    return delay_seconds, peak_idx

def onset_delay(x, y, threshold=0.1):
    """Estimate delay from onset difference."""
    def _find_onset(sig, thresh):
        for i, s in enumerate(sig):
            if abs(s) > thresh * max(abs(v) for v in sig):
                return i
        return 0
    onset_x = _find_onset(x, threshold)
    onset_y = _find_onset(y, threshold)
    return onset_y - onset_x

def peak_delay(x, y):
    """Estimate delay from cross-correlation peak."""
    from dravex.correlation.crosscorr import cross_correlation
    r = cross_correlation(x, y)
    n = len(x)
    peak_idx = max(range(len(r)), key=lambda i: r[i])
    return peak_idx - n + 1

def phase_delay(x, y, sample_rate=44100, max_freq=None):
    """Estimate delay from phase of cross-spectrum."""
    import math
    from dravex.transform.fft import fft
    from dravex.constants import next_power_of_2, TAU, DEFAULT_EPSILON
    n = max(len(x), len(y))
    N = next_power_of_2(n)
    X = fft(list(x) + [0.0] * (N - len(x)))
    Y = fft(list(y) + [0.0] * (N - len(y)))
    delays = []
    for k in range(1, N // 2):
        freq = k * sample_rate / N
        if max_freq and freq > max_freq:
            break
        cross_re = X[k][0]*Y[k][0] + X[k][1]*Y[k][1]
        cross_im = X[k][1]*Y[k][0] - X[k][0]*Y[k][1]
        phase = math.atan2(cross_im, cross_re)
        delay = -phase / (TAU * freq) if freq > DEFAULT_EPSILON else 0
        delays.append(delay)
    if not delays:
        return 0.0
    return sum(delays) / len(delays)

def multi_delay_estimate(x, y, sample_rate=44100, n_estimates=5):
    """Estimate delay using multiple methods and return median."""
    results = []
    d1, _ = gcc_phat(x, y, sample_rate)
    results.append(d1)
    d2 = onset_delay(x, y)
    results.append(d2 / sample_rate)
    d3 = peak_delay(x, y)
    results.append(d3 / sample_rate)
    results.sort()
    return results[len(results) // 2]
