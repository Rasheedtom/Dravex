"""Power spectral density estimation."""
import math
from dravex.transform.fft import rfft
from dravex.constants import DEFAULT_EPSILON

def periodogram(x, sample_rate=44100, window=None):
    """Estimate PSD using the periodogram method.

    Returns (frequencies, power_spectral_density) tuple.
    """
    """Estimate PSD using the periodogram method."""
    n = len(x)
    if window:
        x = [s * w for s, w in zip(x, window)]
        win_power = sum(w * w for w in window) / n
    else:
        win_power = 1.0
    mags = rfft(x)
    psd = [(m * m) / (n * sample_rate * win_power) for m in mags]
    freqs = [i * sample_rate / n for i in range(len(psd))]
    return freqs, psd

def welch_psd(x, frame_size=1024, hop_size=512, sample_rate=44100, window=None):
    """Welch method for PSD estimation (averaged periodograms)."""
    from dravex.window.hann import hann
    if window is None:
        window = hann(frame_size)
    n = len(x)
    n_bins = frame_size // 2 + 1
    avg_psd = [0.0] * n_bins
    n_frames = 0
    pos = 0
    while pos + frame_size <= n:
        chunk = x[pos:pos + frame_size]
        windowed = [s * w for s, w in zip(chunk, window)]
        mags = rfft(windowed)
        for i in range(n_bins):
            avg_psd[i] += mags[i] * mags[i]
        n_frames += 1
        pos += hop_size
    if n_frames > 0:
        win_power = sum(w * w for w in window) / frame_size
        scale = 1.0 / (n_frames * frame_size * sample_rate * win_power) if win_power > 0 else 1.0
        avg_psd = [p * scale for p in avg_psd]
    freqs = [i * sample_rate / frame_size for i in range(n_bins)]
    return freqs, avg_psd

def band_power(psd, freqs, low_hz, high_hz):
    """Compute power in frequency band via trapezoidal integration."""
    total = 0.0
    for i in range(1, len(freqs)):
        if freqs[i-1] >= low_hz and freqs[i] <= high_hz:
            total += (psd[i-1] + psd[i]) * (freqs[i] - freqs[i-1]) / 2
    return total

def total_power(psd, freqs):
    """Total signal power from PSD."""
    total = 0.0
    for i in range(1, len(freqs)):
        total += (psd[i-1] + psd[i]) * (freqs[i] - freqs[i-1]) / 2
    return total

def bartlett_psd(x, n_segments=8, sample_rate=44100):
    """Bartlett method: average of non-overlapping periodograms."""
    n = len(x)
    seg_len = n // n_segments
    n_bins = seg_len // 2 + 1
    avg = [0.0] * n_bins
    for i in range(n_segments):
        segment = x[i * seg_len:(i + 1) * seg_len]
        from dravex.transform.fft import rfft
        mags = rfft(segment)
        for k in range(min(len(mags), n_bins)):
            avg[k] += mags[k] * mags[k]
    avg = [a / n_segments for a in avg]
    freqs = [k * sample_rate / seg_len for k in range(n_bins)]
    return freqs, avg

def spectral_centroid(psd, freqs):
    """Spectral centroid from PSD."""
    total = sum(psd)
    if total < 1e-12:
        return 0.0
    return sum(f * p for f, p in zip(freqs, psd)) / total

def spectral_variance(psd, freqs):
    """Spectral variance (spread)."""
    centroid = spectral_centroid(psd, freqs)
    total = sum(psd)
    if total < 1e-12:
        return 0.0
    return sum(p * (f - centroid)**2 for f, p in zip(freqs, psd)) / total

def dominant_frequency(psd, freqs):
    """Frequency with maximum power."""
    if not psd:
        return 0.0
    idx = max(range(len(psd)), key=lambda i: psd[i])
    return freqs[idx]
