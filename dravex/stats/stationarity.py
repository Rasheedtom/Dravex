"""Signal stationarity tests."""
import math
from dravex.stats.moments import mean, variance

def adf_test_simple(x, max_lag=1):
    """Simplified Augmented Dickey-Fuller test for stationarity."""
    n = len(x)
    if n < max_lag + 2:
        return 0.0, True
    diffs = [x[i] - x[i-1] for i in range(1, n)]
    y = diffs[max_lag:]
    x_lag = [x[i] for i in range(max_lag, n - 1)]
    n_eff = len(y)
    x_mean = sum(x_lag) / n_eff
    y_mean = sum(y) / n_eff
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_lag, y))
    denominator = sum((xi - x_mean)**2 for xi in x_lag)
    if abs(denominator) < 1e-15:
        return 0.0, True
    beta = numerator / denominator
    residuals = [yi - beta * xi for xi, yi in zip(x_lag, y)]
    se = math.sqrt(sum(r*r for r in residuals) / max(1, n_eff - 2))
    se_beta = se / math.sqrt(denominator)
    if se_beta < 1e-15:
        return 0.0, True
    t_stat = beta / se_beta
    critical_5pct = -2.86
    return t_stat, t_stat < critical_5pct

def variance_ratio(x, frame_size=256):
    """Test stationarity via variance ratio across frames."""
    n = len(x)
    frame_vars = []
    pos = 0
    while pos + frame_size <= n:
        frame = x[pos:pos + frame_size]
        frame_vars.append(variance(frame))
        pos += frame_size
    if len(frame_vars) < 2:
        return 1.0
    overall_var = variance(x)
    if overall_var < 1e-15:
        return 1.0
    mean_frame_var = sum(frame_vars) / len(frame_vars)
    return mean_frame_var / overall_var

def rolling_mean_test(x, window_size=256, threshold=0.1):
    """Check if rolling mean stays within threshold of overall mean."""
    overall = mean(x)
    n = len(x)
    deviations = []
    for i in range(n - window_size):
        local = sum(x[i:i + window_size]) / window_size
        deviations.append(abs(local - overall))
    if not deviations:
        return True
    max_dev = max(deviations)
    return max_dev < threshold * (abs(overall) + 1e-10)

def kpss_test_simple(x, n_lags=None):
    """Simplified KPSS test for trend stationarity."""
    import math
    n = len(x)
    if n_lags is None:
        n_lags = int(4 * (n / 100) ** 0.25)
    m = mean(x)
    residuals = [xi - m for xi in x]
    cumsum = [0.0]
    for r in residuals:
        cumsum.append(cumsum[-1] + r)
    s2 = sum(r*r for r in residuals) / n
    if s2 < 1e-15:
        return 0.0, True
    eta = sum(c*c for c in cumsum[1:]) / (n * n * s2)
    critical_5pct = 0.463
    return eta, eta < critical_5pct

def segment_stationarity(x, n_segments=4):
    """Test stationarity by comparing segment statistics."""
    seg_len = len(x) // n_segments
    means = []
    variances = []
    for i in range(n_segments):
        seg = x[i*seg_len:(i+1)*seg_len]
        means.append(mean(seg))
        variances.append(variance(seg))
    mean_range = max(means) - min(means)
    var_range = max(variances) - min(variances) if max(variances) > 0 else 0
    overall_mean = mean(x)
    mean_stable = mean_range < 0.1 * abs(overall_mean + 1e-10)
    var_stable = var_range < 0.5 * max(variances) if max(variances) > 0 else True
    return mean_stable and var_stable

def trend_test(x, significance=0.05):
    """Simple linear trend test via correlation with time."""
    import math
    n = len(x)
    if n < 3:
        return False, 0.0
    time = list(range(n))
    t_mean = (n - 1) / 2
    x_mean = sum(x) / n
    num = sum((t - t_mean) * (xi - x_mean) for t, xi in zip(time, x))
    den_t = sum((t - t_mean)**2 for t in time)
    den_x = sum((xi - x_mean)**2 for xi in x)
    if den_t < 1e-15 or den_x < 1e-15:
        return False, 0.0
    r = num / math.sqrt(den_t * den_x)
    t_stat = r * math.sqrt((n - 2) / (1 - r*r + 1e-15))
    has_trend = abs(t_stat) > 2.0
    return has_trend, r
