"""Discrete Hartley Transform."""
import math
from dravex.constants import TAU

def dht(x):
    """Discrete Hartley Transform. O(N^2)."""
    N = len(x)
    result = []
    for k in range(N):
        s = 0.0
        for n in range(N):
            angle = TAU * k * n / N
            s += x[n] * (math.cos(angle) + math.sin(angle))
        result.append(s)
    return result

def idht(X):
    """Inverse DHT (same as forward DHT scaled by 1/N)."""
    N = len(X)
    result = dht(X)
    return [v / N for v in result]

def dht_to_dft(H):
    """Convert DHT coefficients to DFT."""
    N = len(H)
    dft = []
    for k in range(N):
        re = (H[k] + H[(N - k) % N]) / 2
        im = (H[k] - H[(N - k) % N]) / 2
        dft.append((re, im))
    return dft

def dht_convolution(x, h):
    """Convolution via DHT (real-valued alternative to FFT convolution)."""
    N = max(len(x), len(h))
    x_pad = list(x) + [0.0] * (N - len(x))
    h_pad = list(h) + [0.0] * (N - len(h))
    Hx = dht(x_pad)
    Hh = dht(h_pad)
    product = []
    for k in range(N):
        k_neg = (N - k) % N
        val = (Hx[k] * (Hh[k] + Hh[k_neg]) + Hx[k_neg] * (Hh[k] - Hh[k_neg])) / 2
        product.append(val)
    return idht(product)
