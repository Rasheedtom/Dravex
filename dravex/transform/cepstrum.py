"""Cepstral analysis."""
import math
from dravex.transform.fft import fft, ifft, rfft
from dravex.constants import DEFAULT_EPSILON

def real_cepstrum(x):
    """Real cepstrum: ifft(log(|fft(x)|))."""
    magnitudes = rfft(x)
    log_mags = [math.log(max(m, DEFAULT_EPSILON)) for m in magnitudes]
    n = (len(log_mags) - 1) * 2
    symmetric = log_mags + list(reversed(log_mags[1:-1]))
    result = ifft([(v, 0.0) for v in symmetric])
    return [r for r, _ in result]

def complex_cepstrum(x):
    """Complex cepstrum including phase."""
    X = fft(x)
    log_X = []
    for re, im in X:
        mag = math.sqrt(re*re + im*im)
        phase = math.atan2(im, re)
        log_mag = math.log(max(mag, DEFAULT_EPSILON))
        log_X.append((log_mag * math.cos(phase) - phase * math.sin(phase),
                       log_mag * math.sin(phase) + phase * math.cos(phase)))
    result = ifft(log_X)
    return [(r, i) for r, i in result]

def lifter(cepstrum, cutoff):
    """Apply liftering: zero quefrencies above cutoff."""
    result = list(cepstrum)
    for i in range(cutoff, len(result)):
        result[i] = 0.0
    return result

def cepstral_peak(cepstrum, sample_rate, min_freq=50, max_freq=2000):
    """Find dominant pitch from cepstral peak."""
    min_quefrency = int(sample_rate / max_freq)
    max_quefrency = min(int(sample_rate / min_freq), len(cepstrum))
    if min_quefrency >= max_quefrency:
        return 0.0
    search = cepstrum[min_quefrency:max_quefrency]
    if not search:
        return 0.0
    peak_idx = max(range(len(search)), key=lambda i: search[i])
    quefrency = min_quefrency + peak_idx
    if quefrency == 0:
        return 0.0
    return sample_rate / quefrency

def mfcc_from_cepstrum(cepstrum, n_coeffs=13):
    """Extract MFCC-like coefficients from cepstrum."""
    return cepstrum[:n_coeffs]
