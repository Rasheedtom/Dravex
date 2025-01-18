"""Magnitude spectrum utilities."""
import math
from dravex.transform.fft import rfft
from dravex.constants import amplitude_to_db, DEFAULT_EPSILON

def magnitude_spectrum(x, sample_rate=44100):
    """Compute magnitude spectrum."""
    mags = rfft(x)
    n = len(x)
    freqs = [i * sample_rate / n for i in range(len(mags))]
    return freqs, mags

def log_magnitude(x, sample_rate=44100):
    """Magnitude spectrum in dB."""
    freqs, mags = magnitude_spectrum(x, sample_rate)
    db = [amplitude_to_db(m) for m in mags]
    return freqs, db

def spectral_peaks(magnitudes, n_peaks=5, min_distance=3):
    """Find peaks in magnitude spectrum."""
    peaks = []
    n = len(magnitudes)
    for i in range(1, n - 1):
        if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
            peaks.append((i, magnitudes[i]))
    peaks.sort(key=lambda p: p[1], reverse=True)
    result = []
    for idx, mag in peaks:
        too_close = False
        for prev_idx, _ in result:
            if abs(idx - prev_idx) < min_distance:
                too_close = True
                break
        if not too_close:
            result.append((idx, mag))
        if len(result) >= n_peaks:
            break
    return result

def spectral_contrast(magnitudes, n_bands=6):
    """Spectral contrast: difference between peaks and valleys per band."""
    n = len(magnitudes)
    band_size = n // n_bands
    contrasts = []
    for b in range(n_bands):
        start = b * band_size
        end = min(start + band_size, n)
        band = magnitudes[start:end]
        if band:
            sorted_band = sorted(band)
            top = sum(sorted_band[-max(1,len(sorted_band)//4):]) / max(1, len(sorted_band)//4)
            bottom = sum(sorted_band[:max(1,len(sorted_band)//4)]) / max(1, len(sorted_band)//4)
            contrasts.append(amplitude_to_db(max(top, DEFAULT_EPSILON)) - amplitude_to_db(max(bottom, DEFAULT_EPSILON)))
        else:
            contrasts.append(0.0)
    return contrasts
