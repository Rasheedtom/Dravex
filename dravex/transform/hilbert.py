"""Hilbert transform for analytic signal construction."""
import math
from dravex.transform.fft import fft, ifft
from dravex.constants import next_power_of_2

def hilbert_transform(x):
    """Compute the Hilbert transform of a real signal."""
    n = len(x)
    N = next_power_of_2(n)
    padded = list(x) + [0.0] * (N - n)
    X = fft(padded)
    H = [(0.0, 0.0)] * N
    H[0] = X[0]
    for k in range(1, N // 2):
        H[k] = (2 * X[k][0], 2 * X[k][1])
    if N > 1:
        H[N // 2] = X[N // 2]
    h = ifft(H)
    return [h[i][1] for i in range(n)]

def analytic_signal(x):
    """Construct analytic signal: z(t) = x(t) + j*H{x(t)}.

    The analytic signal has zero negative-frequency content.
    """
    """Construct analytic signal: x + j*hilbert(x)."""
    ht = hilbert_transform(x)
    return [(x[i], ht[i]) for i in range(len(x))]

def instantaneous_amplitude(x):
    """Envelope via analytic signal."""
    a = analytic_signal(x)
    return [math.sqrt(re*re + im*im) for re, im in a]

def instantaneous_phase(x):
    """Instantaneous phase via analytic signal."""
    a = analytic_signal(x)
    return [math.atan2(im, re) for re, im in a]

def instantaneous_frequency(x, sample_rate=44100):
    """Instantaneous frequency from phase derivative."""
    phase = instantaneous_phase(x)
    freq = [0.0]
    for i in range(1, len(phase)):
        dp = phase[i] - phase[i-1]
        while dp > math.pi:
            dp -= 2 * math.pi
        while dp < -math.pi:
            dp += 2 * math.pi
        freq.append(dp * sample_rate / (2 * math.pi))
    return freq
