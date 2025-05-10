"""Autocorrelation computation."""

def autocorrelation(x, max_lag=None):
    """Compute unnormalized autocorrelation function.

    For normalized version, use normalized_autocorrelation().
    """
    """Compute autocorrelation function."""
    n = len(x)
    if max_lag is None:
        max_lag = n
    result = []
    for lag in range(min(max_lag, n)):
        s = 0.0
        for i in range(n - lag):
            s += x[i] * x[i + lag]
        result.append(s)
    return result

def normalized_autocorrelation(x, max_lag=None):
    """Autocorrelation normalized by zero-lag value."""
    r = autocorrelation(x, max_lag)
    if not r or r[0] == 0:
        return r
    return [ri / r[0] for ri in r]

def autocorrelation_fft(x):
    """Fast autocorrelation via FFT."""
    from dravex.convolve.fast_conv import autocorrelation_fft as _acf
    return _acf(x)

def partial_autocorrelation(x, max_lag=20):
    """Partial autocorrelation function (PACF) via Levinson-Durbin."""
    r = autocorrelation(x, max_lag + 1)
    if not r or r[0] == 0:
        return [0.0] * max_lag
    pacf = []
    a = [0.0] * (max_lag + 1)
    a[1] = r[1] / r[0]
    pacf.append(a[1])
    for k in range(2, max_lag + 1):
        num = r[k]
        for j in range(1, k):
            num -= a[j] * r[k - j]
        denom = r[0]
        for j in range(1, k):
            denom -= a[j] * r[j]
        if abs(denom) < 1e-15:
            pacf.append(0.0)
            continue
        a_new = [0.0] * (max_lag + 1)
        a_new[k] = num / denom
        for j in range(1, k):
            a_new[j] = a[j] - a_new[k] * a[k - j]
        a = a_new
        pacf.append(a[k])
    return pacf
