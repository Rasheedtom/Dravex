"""Mel-frequency cepstral coefficients."""
import math
from dravex.constants import hz_to_mel, mel_to_hz, DEFAULT_EPSILON
from dravex.transform.fft import rfft
from dravex.transform.dct import dct_ii

def mfcc(x, sample_rate=44100, n_mfcc=13, n_filters=26, n_fft=512):
    """Compute MFCCs from signal samples."""
    magnitudes = rfft(x[:n_fft] + [0.0] * max(0, n_fft - len(x)))
    mel_filters = mel_filterbank(n_filters, n_fft, sample_rate)
    mel_energies = []
    for filt in mel_filters:
        energy = sum(m * f for m, f in zip(magnitudes, filt))
        mel_energies.append(max(energy, DEFAULT_EPSILON))
    log_energies = [math.log(e) for e in mel_energies]
    coeffs = dct_ii(log_energies)
    return coeffs[:n_mfcc]

def mel_filterbank(n_filters, n_fft, sample_rate):
    """Generate triangular mel-spaced filterbank.

    The mel scale approximates human auditory perception,
    providing higher resolution at lower frequencies.
    """
    """Generate triangular mel-spaced filterbank."""
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mel_points = [low_mel + i * (high_mel - low_mel) / (n_filters + 1) for i in range(n_filters + 2)]
    hz_points = [mel_to_hz(m) for m in mel_points]
    bin_points = [int(f * n_fft / sample_rate) for f in hz_points]
    n_bins = n_fft // 2 + 1
    filters = []
    for i in range(n_filters):
        filt = [0.0] * n_bins
        for j in range(n_bins):
            if bin_points[i] <= j < bin_points[i+1]:
                denom = bin_points[i+1] - bin_points[i]
                filt[j] = (j - bin_points[i]) / denom if denom > 0 else 0
            elif bin_points[i+1] <= j < bin_points[i+2]:
                denom = bin_points[i+2] - bin_points[i+1]
                filt[j] = (bin_points[i+2] - j) / denom if denom > 0 else 0
        filters.append(filt)
    return filters

def delta_mfcc(mfcc_frames, width=2):
    """Compute delta (velocity) coefficients from MFCC sequence."""
    n = len(mfcc_frames)
    deltas = []
    denom = 2 * sum(t * t for t in range(1, width + 1))
    if denom == 0:
        return [list(f) for f in mfcc_frames]
    for i in range(n):
        delta = [0.0] * len(mfcc_frames[i])
        for t in range(1, width + 1):
            prev_idx = max(0, i - t)
            next_idx = min(n - 1, i + t)
            for j in range(len(delta)):
                delta[j] += t * (mfcc_frames[next_idx][j] - mfcc_frames[prev_idx][j])
        deltas.append([d / denom for d in delta])
    return deltas

def mfcc_full(x, sample_rate=44100, n_mfcc=13, n_filters=26, frame_size=512, hop_size=256):
    """Compute MFCCs for each frame of a signal."""
    from dravex.window.hann import hann
    n = len(x)
    window = hann(frame_size)
    all_mfcc = []
    pos = 0
    while pos + frame_size <= n:
        frame = x[pos:pos + frame_size]
        windowed = [s * w for s, w in zip(frame, window)]
        coeffs = mfcc(windowed, sample_rate, n_mfcc, n_filters, frame_size)
        all_mfcc.append(coeffs)
        pos += hop_size
    return all_mfcc

def mfcc_features(x, sample_rate=44100, n_mfcc=13):
    """Compute MFCC + delta + delta-delta features."""
    frames = mfcc_full(x, sample_rate, n_mfcc)
    if not frames:
        return []
    deltas = delta_mfcc(frames, 2)
    delta_deltas = delta_mfcc(deltas, 2)
    features = []
    for mf, d, dd in zip(frames, deltas, delta_deltas):
        features.append(mf + d + dd)
    return features

def lifter_mfcc(mfcc_coeffs, lift=22):
    """Apply sinusoidal liftering to MFCCs."""
    import math
    n = len(mfcc_coeffs)
    lifter_vals = [1 + (lift / 2) * math.sin(math.pi * i / lift) for i in range(n)]
    return [c * l for c, l in zip(mfcc_coeffs, lifter_vals)]
