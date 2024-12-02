"""Standalone real cepstrum with additional utilities."""
import math
from dravex.transform.fft import fft, ifft, rfft
from dravex.constants import DEFAULT_EPSILON

def power_cepstrum(x):
    """Power cepstrum: |ifft(log(|fft(x)|^2))|^2."""
    mags = rfft(x)
    log_power = [math.log(max(m * m, DEFAULT_EPSILON)) for m in mags]
    n = (len(log_power) - 1) * 2
    symmetric = log_power + list(reversed(log_power[1:-1]))
    result = ifft([(v, 0.0) for v in symmetric])
    return [r * r + i * i for r, i in result]

def cepstral_smoothing(log_spectrum, n_lifter=30):
    """Smooth log spectrum using cepstral liftering."""
    n = len(log_spectrum)
    symmetric = log_spectrum + list(reversed(log_spectrum[1:-1]))
    from dravex.transform.fft import fft, ifft
    cepstrum = ifft([(v, 0.0) for v in symmetric])
    for i in range(n_lifter, len(cepstrum) - n_lifter):
        cepstrum[i] = (0.0, 0.0)
    smoothed = fft([r for r, _ in cepstrum])
    return [r for r, _ in smoothed[:n]]

def homomorphic_filter(x, cutoff_quefrency=30):
    """Homomorphic filtering to separate source and system."""
    mags = rfft(x)
    log_mags = [math.log(max(m, DEFAULT_EPSILON)) for m in mags]
    smoothed = cepstral_smoothing(log_mags, cutoff_quefrency)
    system_response = [math.exp(s) for s in smoothed]
    source = [m / max(sr, DEFAULT_EPSILON) for m, sr in zip(mags, system_response)]
    return system_response, source

def quefrency_to_frequency(quefrency, sample_rate):
    """Convert quefrency (cepstral domain) to frequency."""
    if quefrency <= 0:
        return 0.0
    return sample_rate / quefrency

def frequency_to_quefrency(frequency, sample_rate):
    """Convert frequency to quefrency."""
    if frequency <= 0:
        return 0.0
    return sample_rate / frequency
