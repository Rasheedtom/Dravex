"""Spectral subtraction denoising."""
import math
from dravex.transform.fft import fft, ifft, rfft
from dravex.transform.stft import stft, istft
from dravex.constants import DEFAULT_EPSILON

def spectral_subtraction(noisy, noise_estimate, frame_size=1024, hop_size=256, alpha=2.0, beta=0.01):
    """Denoise via spectral subtraction.

    Alpha controls over-subtraction factor; beta is the spectral floor.
    """
    """Denoise via spectral subtraction."""
    from dravex.window.hann import hann
    window = hann(frame_size)
    noisy_frames = stft(noisy, frame_size, hop_size, window)
    noise_frames = stft(noise_estimate, frame_size, hop_size, window) if len(noise_estimate) >= frame_size else None
    if noise_frames:
        noise_mag = [sum(m) / len(noise_frames) for m in zip(*[f[0] for f in noise_frames])]
    else:
        noise_mag = [0.0] * len(noisy_frames[0][0]) if noisy_frames else []
    denoised_frames = []
    for mags, phases in noisy_frames:
        clean_mags = []
        for m, n_m in zip(mags, noise_mag):
            subtracted = m**alpha - n_m**alpha
            if subtracted > 0:
                clean_mags.append(subtracted ** (1.0 / alpha))
            else:
                clean_mags.append(beta * m)
        denoised_frames.append((clean_mags, phases))
    return istft(denoised_frames, frame_size, hop_size, window)

def wiener_filter_denoise(noisy, noise_power_estimate, frame_size=1024, hop_size=256):
    """Wiener filter denoising in frequency domain."""
    from dravex.window.hann import hann
    window = hann(frame_size)
    frames = stft(noisy, frame_size, hop_size, window)
    denoised = []
    for mags, phases in frames:
        clean_mags = []
        for m in mags:
            sig_power = max(0, m * m - noise_power_estimate)
            gain = sig_power / (sig_power + noise_power_estimate) if sig_power + noise_power_estimate > DEFAULT_EPSILON else 0
            clean_mags.append(m * gain)
        denoised.append((clean_mags, phases))
    return istft(denoised, frame_size, hop_size, window)

def soft_threshold_denoise(x, frame_size=1024, hop_size=256, threshold_factor=2.0):
    """Denoise via soft thresholding in STFT domain."""
    import math
    from dravex.transform.stft import stft, istft
    from dravex.window.hann import hann
    window = hann(frame_size)
    frames = stft(x, frame_size, hop_size, window)
    all_mags = []
    for mags, _ in frames:
        all_mags.extend(mags)
    noise_floor = sorted(all_mags)[len(all_mags) // 4]
    threshold = noise_floor * threshold_factor
    denoised = []
    for mags, phases in frames:
        new_mags = []
        for m in mags:
            if m > threshold:
                new_mags.append(m - threshold)
            else:
                new_mags.append(0.0)
        denoised.append((new_mags, phases))
    return istft(denoised, frame_size, hop_size, window)

def median_denoise(x, frame_size=5):
    """Denoise using median filter."""
    from dravex.filter.median import median_filter
    return median_filter(x, frame_size)
