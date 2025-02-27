"""Fast convolution utilities and benchmarking."""
from dravex.convolve.direct import convolve as direct_convolve
from dravex.convolve.overlap_add import fft_convolve

def auto_convolve(x, h, threshold=64):
    """Choose direct or FFT convolution based on kernel size."""
    if len(h) <= threshold:
        return direct_convolve(x, h)
    return fft_convolve(x, h)

def circular_convolve(x, h):
    """Circular (periodic) convolution."""
    n = max(len(x), len(h))
    x_ext = list(x) + [0.0] * (n - len(x))
    h_ext = list(h) + [0.0] * (n - len(h))
    from dravex.transform.fft import fft, ifft
    X = fft(x_ext)
    H = fft(h_ext)
    Y = [(xr*hr - xi*hi, xr*hi + xi*hr) for (xr, xi), (hr, hi) in zip(X, H)]
    y = ifft(Y)
    return [y[i][0] for i in range(n)]

def autocorrelation_fft(x):
    """Autocorrelation via FFT (faster than direct for long signals)."""
    from dravex.transform.fft import fft, ifft
    from dravex.constants import next_power_of_2
    n = len(x)
    N = next_power_of_2(2 * n)
    X = fft(list(x) + [0.0] * (N - n))
    power = [(xr*xr + xi*xi, 0.0) for xr, xi in X]
    r = ifft(power)
    return [r[i][0] for i in range(n)]

def conv_matrix(h, n):
    """Build convolution matrix for kernel h applied to signal of length n."""
    m = len(h)
    rows = n + m - 1
    matrix = [[0.0] * n for _ in range(rows)]
    for i in range(rows):
        for j in range(m):
            col = i - j
            if 0 <= col < n:
                matrix[i][col] = h[j]
    return matrix
