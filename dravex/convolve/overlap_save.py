"""Overlap-save fast convolution method."""
from dravex.transform.fft import fft, ifft
from dravex.constants import next_power_of_2

def overlap_save(x, h, block_size=None):
    """Overlap-save convolution."""
    m = len(h)
    if block_size is None:
        block_size = max(4 * m, 256)
    N = next_power_of_2(block_size)
    if N < m:
        N = next_power_of_2(m)
    overlap = m - 1
    step = N - overlap
    h_padded = list(h) + [0.0] * (N - m)
    H = fft(h_padded)
    x_ext = [0.0] * overlap + list(x) + [0.0] * N
    out_len = len(x) + m - 1
    y = [0.0] * out_len
    pos = 0
    out_pos = 0
    while out_pos < out_len:
        chunk = x_ext[pos:pos + N]
        if len(chunk) < N:
            chunk = chunk + [0.0] * (N - len(chunk))
        X = fft(chunk)
        Y = [(xr*hr - xi*hi, xr*hi + xi*hr) for (xr, xi), (hr, hi) in zip(X, H)]
        y_block = ifft(Y)
        for i in range(overlap, N):
            idx = out_pos + i - overlap
            if idx < out_len:
                y[idx] = y_block[i][0]
        pos += step
        out_pos += step
    return y[:out_len]

def partitioned_convolution(x, h, partition_size=256):
    """Partitioned convolution for very long impulse responses."""
    from dravex.transform.fft import fft, ifft
    from dravex.constants import next_power_of_2
    m = len(h)
    n_partitions = (m + partition_size - 1) // partition_size
    N = next_power_of_2(2 * partition_size)
    h_parts = []
    for p in range(n_partitions):
        start = p * partition_size
        end = min(start + partition_size, m)
        part = list(h[start:end]) + [0.0] * (N - (end - start))
        h_parts.append(fft(part))
    n = len(x)
    out_len = n + m - 1
    y = [0.0] * out_len
    fdl = [[(0.0, 0.0)] * N for _ in range(n_partitions)]
    pos = 0
    while pos < n:
        chunk = x[pos:pos + partition_size]
        chunk_padded = list(chunk) + [0.0] * (N - len(chunk))
        X = fft(chunk_padded)
        for p in range(n_partitions - 1, 0, -1):
            fdl[p] = fdl[p - 1]
        fdl[0] = X
        Y = [(0.0, 0.0)] * N
        for p in range(n_partitions):
            for k in range(N):
                xr, xi = fdl[p][k]
                hr, hi = h_parts[p][k]
                yr, yi = Y[k]
                Y[k] = (yr + xr*hr - xi*hi, yi + xr*hi + xi*hr)
        y_block = ifft(Y)
        for i in range(partition_size):
            idx = pos + i
            if idx < out_len:
                y[idx] += y_block[i][0]
        pos += partition_size
    return y[:out_len]
