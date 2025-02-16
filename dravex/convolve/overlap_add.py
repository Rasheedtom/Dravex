"""Overlap-add fast convolution method."""
from dravex.transform.fft import fft, ifft
from dravex.constants import next_power_of_2

def overlap_add(x, h, block_size=None):
    """Overlap-add convolution.

    Overall complexity: O(N log N) where N = len(x) + len(h).
    """
    """Overlap-add convolution. O(N log N) overall."""
    n = len(x)
    m = len(h)
    if block_size is None:
        block_size = max(m, 256)
    N = next_power_of_2(block_size + m - 1)
    h_padded = list(h) + [0.0] * (N - m)
    H = fft(h_padded)
    out_len = n + m - 1
    y = [0.0] * out_len
    pos = 0
    while pos < n:
        chunk = x[pos:pos + block_size]
        chunk_padded = list(chunk) + [0.0] * (N - len(chunk))
        X = fft(chunk_padded)
        Y = [(xr*hr - xi*hi, xr*hi + xi*hr) for (xr, xi), (hr, hi) in zip(X, H)]
        y_block = ifft(Y)
        for i in range(min(N, out_len - pos)):
            y[pos + i] += y_block[i][0]
        pos += block_size
    return y[:out_len]

def fft_convolve(x, h):
    """Full FFT-based convolution."""
    n = len(x) + len(h) - 1
    N = next_power_of_2(n)
    X = fft(list(x) + [0.0] * (N - len(x)))
    H = fft(list(h) + [0.0] * (N - len(h)))
    Y = [(xr*hr - xi*hi, xr*hi + xi*hr) for (xr, xi), (hr, hi) in zip(X, H)]
    y = ifft(Y)
    return [y[i][0] for i in range(n)]
