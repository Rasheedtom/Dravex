"""Direct convolution implementation."""
def convolve(x, h):
    """Direct linear convolution. O(N*M)."""
    n = len(x)
    m = len(h)
    if n == 0 or m == 0:
        return []
    out_len = n + m - 1
    y = [0.0] * out_len
    for i in range(n):
        for j in range(m):
            y[i + j] += x[i] * h[j]
    return y

def convolve_same(x, h):
    """Convolution with output length equal to input."""
    full = convolve(x, h)
    offset = (len(h) - 1) // 2
    return full[offset:offset + len(x)]

def convolve_valid(x, h):
    """Convolution keeping only fully-overlapping region."""
    full = convolve(x, h)
    m = len(h) - 1
    return full[m:len(full) - m] if len(full) > 2 * m else full

def correlate(x, h):
    """Cross-correlation via convolution with reversed kernel."""
    return convolve(x, list(reversed(h)))

def running_average(x, window_size):
    """Running average as convolution with uniform kernel."""
    kernel = [1.0 / window_size] * window_size
    return convolve_same(x, kernel)

def convolve_symmetric(x, h):
    """Convolution exploiting symmetric kernel (saves half the multiplies)."""
    m = len(h)
    n = len(x)
    if m < 3:
        return convolve(x, h)
    half = m // 2
    is_sym = all(abs(h[i] - h[m-1-i]) < 1e-12 for i in range(half))
    if not is_sym:
        return convolve(x, h)
    out_len = n + m - 1
    y = [0.0] * out_len
    for i in range(n):
        y[i + half] += x[i] * h[half]
        for j in range(half):
            y[i + j] += x[i] * h[j]
            y[i + m - 1 - j] += x[i] * h[j]
    return y

def deconvolve_td(y, h):
    """Time-domain deconvolution (inverse filtering by long division)."""
    n = len(y)
    m = len(h)
    if m == 0 or h[0] == 0:
        return list(y)
    x = [0.0] * (n - m + 1)
    remainder = list(y)
    for i in range(len(x)):
        x[i] = remainder[i] / h[0]
        for j in range(m):
            if i + j < n:
                remainder[i + j] -= x[i] * h[j]
    return x
