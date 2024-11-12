"""Fast Fourier Transform (Cooley-Tukey radix-2)."""
import math
from dravex.constants import PI, TAU, next_power_of_2

def fft(x):
    """Radix-2 FFT. Input length must be power of 2."""
    n = len(x)
    if n <= 1:
        if not x:
            return [(0.0, 0.0)]
        if isinstance(x[0], tuple):
            return [x[0]]
        return [(x[0], 0.0)]
    if isinstance(x[0], tuple):
        if n & (n - 1):
            padded = list(x) + [(0.0, 0.0)] * (next_power_of_2(n) - n)
            return fft(padded)
        real = [r for r, i in x]
        imag = [i for r, i in x]
    else:
        if n & (n - 1):
            padded = list(x) + [0.0] * (next_power_of_2(n) - n)
            return fft(padded)
        real = list(x)
        imag = [0.0] * n
    _bit_reverse(real, imag, n)
    m = 2
    while m <= n:
        half = m // 2
        twiddle_re = math.cos(TAU / m)
        twiddle_im = -math.sin(TAU / m)
        for k in range(0, n, m):
            tw_re, tw_im = 1.0, 0.0
            for j in range(half):
                idx1 = k + j
                idx2 = k + j + half
                tr = tw_re * real[idx2] - tw_im * imag[idx2]
                ti = tw_re * imag[idx2] + tw_im * real[idx2]
                real[idx2] = real[idx1] - tr
                imag[idx2] = imag[idx1] - ti
                real[idx1] += tr
                imag[idx1] += ti
                tw_re, tw_im = tw_re * twiddle_re - tw_im * twiddle_im, tw_re * twiddle_im + tw_im * twiddle_re
        m <<= 1
    return list(zip(real, imag))

def ifft(X):
    """Inverse FFT."""
    n = len(X)
    conj_X = [(re, -im) for re, im in X]
    Y = fft(conj_X)
    return [(yr / n, -yi / n) for yr, yi in Y]

def rfft(x):
    """Real-input FFT returning magnitude spectrum."""
    result = fft(x)
    n = len(result) // 2 + 1
    return [math.sqrt(re*re + im*im) for re, im in result[:n]]

def rfft_phase(x):
    """Real-input FFT returning phase spectrum."""
    result = fft(x)
    n = len(result) // 2 + 1
    return [math.atan2(im, re) for re, im in result[:n]]

def _bit_reverse(real, imag, n):
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]

def magnitude_phase(fft_result):
    """Split FFT result into magnitude and phase arrays."""
    mags = [math.sqrt(re*re + im*im) for re, im in fft_result]
    phases = [math.atan2(im, re) for re, im in fft_result]
    return mags, phases

def fft_shift(x):
    """Shift zero-frequency component to center."""
    n = len(x)
    half = n // 2
    return x[half:] + x[:half]

def zero_pad(x, target_length):
    """Zero-pad signal to target length."""
    if len(x) >= target_length:
        return list(x)
    return list(x) + [0.0] * (target_length - len(x))

def power_spectrum(x):
    """Compute power spectrum |X(k)|^2."""
    result = fft(x)
    return [re*re + im*im for re, im in result]

def cross_spectrum(x, y):
    """Compute cross-spectrum X(k) * conj(Y(k))."""
    X = fft(x)
    Y = fft(y)
    return [(xr*yr + xi*yi, xi*yr - xr*yi) for (xr, xi), (yr, yi) in zip(X, Y)]

def spectral_flatness(magnitudes):
    """Spectral flatness (Wiener entropy)."""
    import math
    n = len(magnitudes)
    if n == 0:
        return 0.0
    powers = [m * m for m in magnitudes if m > 1e-12]
    if not powers:
        return 0.0
    log_mean = sum(math.log(p) for p in powers) / len(powers)
    arith_mean = sum(powers) / len(powers)
    if arith_mean < 1e-12:
        return 0.0
    return math.exp(log_mean) / arith_mean
