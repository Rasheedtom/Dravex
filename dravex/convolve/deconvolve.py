"""Deconvolution / inverse filtering."""
from dravex.transform.fft import fft, ifft
from dravex.constants import next_power_of_2, DEFAULT_EPSILON

def wiener_deconvolve(y, h, noise_power=0.01):
    """Wiener deconvolution: estimate x from y = x * h + n."""
    n = len(y)
    m = len(h)
    N = next_power_of_2(max(n, m))
    Y = fft(list(y) + [0.0] * (N - n))
    H = fft(list(h) + [0.0] * (N - m))
    X = []
    for (yr, yi), (hr, hi) in zip(Y, H):
        h_mag_sq = hr * hr + hi * hi
        if h_mag_sq < DEFAULT_EPSILON:
            X.append((0.0, 0.0))
        else:
            wiener = h_mag_sq / (h_mag_sq + noise_power)
            conj_hr = hr
            conj_hi = -hi
            xr = (yr * conj_hr - yi * conj_hi) / h_mag_sq * wiener
            xi = (yr * conj_hi + yi * conj_hr) / h_mag_sq * wiener
            X.append((xr, xi))
    x = ifft(X)
    return [x[i][0] for i in range(n)]

def spectral_division(y, h, regularization=0.001):
    """Simple spectral division deconvolution."""
    N = next_power_of_2(max(len(y), len(h)))
    Y = fft(list(y) + [0.0] * (N - len(y)))
    H = fft(list(h) + [0.0] * (N - len(h)))
    X = []
    for (yr, yi), (hr, hi) in zip(Y, H):
        denom = hr*hr + hi*hi + regularization
        xr = (yr*hr + yi*hi) / denom
        xi = (yi*hr - yr*hi) / denom
        X.append((xr, xi))
    x = ifft(X)
    return [x[i][0] for i in range(len(y))]

def inverse_filter(h, length, regularization=0.01):
    """Compute inverse filter of h."""
    N = next_power_of_2(max(len(h), length))
    H = fft(list(h) + [0.0] * (N - len(h)))
    G = []
    for hr, hi in H:
        denom = hr*hr + hi*hi + regularization
        G.append((hr / denom, -hi / denom))
    g = ifft(G)
    return [g[i][0] for i in range(length)]

def regularized_deconvolution(y, h, lambda_reg=0.01):
    """Tikhonov-regularized deconvolution."""
    from dravex.transform.fft import fft, ifft
    from dravex.constants import next_power_of_2
    n = len(y)
    N = next_power_of_2(max(n, len(h)))
    Y = fft(list(y) + [0.0] * (N - n))
    H = fft(list(h) + [0.0] * (N - len(h)))
    X = []
    for (yr, yi), (hr, hi) in zip(Y, H):
        h_mag_sq = hr*hr + hi*hi
        denom = h_mag_sq + lambda_reg
        xr = (yr*hr + yi*hi) / denom
        xi = (yi*hr - yr*hi) / denom
        X.append((xr, xi))
    x = ifft(X)
    return [x[i][0] for i in range(n)]

def lucy_richardson(y, h, n_iter=50):
    """Lucy-Richardson iterative deconvolution."""
    from dravex.convolve.direct import convolve
    n = len(y)
    estimate = [1.0 / n] * n
    h_flip = list(reversed(h))
    for _ in range(n_iter):
        blurred = convolve(estimate, h)[:n]
        ratio = [y[i] / max(blurred[i], 1e-12) for i in range(n)]
        correction = convolve(ratio, h_flip)[:n]
        estimate = [max(1e-12, estimate[i] * correction[i]) for i in range(n)]
    return estimate
