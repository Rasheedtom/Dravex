"""Onset detection for rhythmic/temporal event finding."""
import math
from dravex.constants import DEFAULT_EPSILON

def spectral_flux(spectrogram_frames):
    """Onset detection via spectral flux.

    Measures half-wave rectified spectral difference between
    consecutive frames to detect transient events.
    """
    """Onset detection via spectral flux."""
    flux = [0.0]
    for i in range(1, len(spectrogram_frames)):
        curr = spectrogram_frames[i]
        prev = spectrogram_frames[i-1]
        diff = 0.0
        for c, p in zip(curr, prev):
            d = c - p
            if d > 0:
                diff += d * d
        flux.append(math.sqrt(diff))
    return flux

def energy_onset(signal_samples, frame_size=1024, hop_size=512):
    """Onset detection via energy difference."""
    n = len(signal_samples)
    energies = []
    pos = 0
    while pos + frame_size <= n:
        e = sum(s*s for s in signal_samples[pos:pos+frame_size]) / frame_size
        energies.append(e)
        pos += hop_size
    onset = [0.0]
    for i in range(1, len(energies)):
        diff = energies[i] - energies[i-1]
        onset.append(max(0, diff))
    return onset

def pick_onsets(onset_function, threshold=None, min_distance=5):
    """Select onset times from onset detection function."""
    if threshold is None:
        threshold = sum(onset_function) / max(1, len(onset_function)) * 1.5
    peaks = []
    for i in range(1, len(onset_function) - 1):
        if onset_function[i] > threshold:
            if onset_function[i] >= onset_function[i-1] and onset_function[i] >= onset_function[i+1]:
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
    return peaks

def hfc_onset(spectrogram_frames):
    """High-Frequency Content onset detection."""
    hfc = []
    for frame in spectrogram_frames:
        weighted = sum((k+1) * m * m for k, m in enumerate(frame))
        hfc.append(weighted)
    diff = [0.0]
    for i in range(1, len(hfc)):
        diff.append(max(0, hfc[i] - hfc[i-1]))
    return diff

def complex_domain_onset(stft_frames):
    """Complex-domain onset detection function."""
    import math
    odf = [0.0]
    for i in range(1, len(stft_frames)):
        curr_mags, curr_phases = stft_frames[i]
        prev_mags, prev_phases = stft_frames[i-1]
        diff = 0.0
        for k in range(len(curr_mags)):
            expected_phase = prev_phases[k]
            if i >= 2:
                expected_phase = 2 * stft_frames[i-1][1][k] - stft_frames[i-2][1][k]
            pred_re = prev_mags[k] * math.cos(expected_phase)
            pred_im = prev_mags[k] * math.sin(expected_phase)
            curr_re = curr_mags[k] * math.cos(curr_phases[k])
            curr_im = curr_mags[k] * math.sin(curr_phases[k])
            dr = curr_re - pred_re
            di = curr_im - pred_im
            diff += math.sqrt(dr*dr + di*di)
        odf.append(diff)
    return odf

def onset_strength(signal_samples, sample_rate=44100, frame_size=2048, hop_size=512):
    """Compute onset strength envelope."""
    from dravex.transform.stft import spectrogram
    spec = spectrogram(signal_samples, frame_size, hop_size)
    strength = [0.0]
    for i in range(1, len(spec)):
        diff = sum(max(0, spec[i][k] - spec[i-1][k]) for k in range(len(spec[i])))
        strength.append(diff)
    return strength
