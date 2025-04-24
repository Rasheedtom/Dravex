"""Signal-to-noise ratio estimation and manipulation."""
import math
from dravex.constants import DEFAULT_EPSILON, power_to_db, db_to_power

def estimate_snr(signal, noise):
    """Estimate SNR from separate signal and noise."""
    sig_power = sum(s * s for s in signal) / len(signal) if signal else 0
    noise_power = sum(n * n for n in noise) / len(noise) if noise else DEFAULT_EPSILON
    if noise_power < DEFAULT_EPSILON:
        return 100.0
    return power_to_db(sig_power / noise_power)

def estimate_snr_single(noisy, frame_size=1024):
    """Estimate SNR from single noisy signal (silence-based)."""
    n = len(noisy)
    frame_energies = []
    pos = 0
    while pos + frame_size <= n:
        e = sum(s * s for s in noisy[pos:pos + frame_size]) / frame_size
        frame_energies.append(e)
        pos += frame_size
    if not frame_energies:
        return 0.0
    sorted_e = sorted(frame_energies)
    noise_est = sorted_e[len(sorted_e) // 10] if len(sorted_e) > 10 else sorted_e[0]
    signal_est = sorted_e[-1]
    if noise_est < DEFAULT_EPSILON:
        return 100.0
    return power_to_db(signal_est / noise_est)

def add_noise_at_snr(signal, target_snr_db, rng=None):
    """Add white noise at specified SNR."""
    from dravex.noise.generators import gaussian_noise
    sig_power = sum(s * s for s in signal) / len(signal)
    noise_power = sig_power / db_to_power(target_snr_db)
    noise_std = math.sqrt(noise_power)
    noise = gaussian_noise(len(signal), 0, noise_std, rng)
    return [s + n for s, n in zip(signal, noise)]

def segmental_snr(clean, noisy, frame_size=256):
    """Compute segmental SNR (average SNR per frame)."""
    n = min(len(clean), len(noisy))
    snrs = []
    pos = 0
    while pos + frame_size <= n:
        sig_e = sum(clean[pos + i]**2 for i in range(frame_size))
        noise_e = sum((clean[pos + i] - noisy[pos + i])**2 for i in range(frame_size))
        if noise_e > DEFAULT_EPSILON:
            snrs.append(power_to_db(sig_e / noise_e))
        pos += frame_size
    return sum(snrs) / len(snrs) if snrs else 0.0

def peak_snr(clean, noisy):
    """Peak SNR (PSNR)."""
    import math
    n = min(len(clean), len(noisy))
    if n == 0:
        return 0.0
    peak = max(abs(s) for s in clean)
    mse = sum((clean[i] - noisy[i])**2 for i in range(n)) / n
    if mse < 1e-15:
        return 100.0
    return 20 * math.log10(peak / math.sqrt(mse))

def si_sdr(clean, estimate):
    """Scale-Invariant Signal-to-Distortion Ratio."""
    import math
    n = min(len(clean), len(estimate))
    s_target_num = sum(clean[i] * estimate[i] for i in range(n))
    s_target_den = sum(clean[i] * clean[i] for i in range(n))
    if s_target_den < 1e-15:
        return 0.0
    alpha = s_target_num / s_target_den
    target = [alpha * clean[i] for i in range(n)]
    noise = [estimate[i] - target[i] for i in range(n)]
    target_power = sum(t * t for t in target)
    noise_power = sum(e * e for e in noise)
    if noise_power < 1e-15:
        return 100.0
    return 10 * math.log10(target_power / noise_power)

def snr_per_band(clean, noisy, n_bands=4, sample_rate=44100):
    """SNR broken down by frequency bands."""
    import math
    from dravex.filter.fir import fir_bandpass, apply_fir
    band_width = (sample_rate // 2) / n_bands
    results = []
    for b in range(n_bands):
        low = max(20, b * band_width)
        high = min((b + 1) * band_width, sample_rate // 2 - 1)
        h = fir_bandpass(64, low, high, sample_rate)
        clean_band = apply_fir(clean, h)
        noisy_band = apply_fir(noisy, h)
        noise_band = [n - c for c, n in zip(clean_band, noisy_band)]
        sig_p = sum(s*s for s in clean_band[100:]) / max(1, len(clean_band) - 100)
        noi_p = sum(s*s for s in noise_band[100:]) / max(1, len(noise_band) - 100)
        if noi_p > 1e-15:
            results.append(10 * math.log10(sig_p / noi_p))
        else:
            results.append(100.0)
    return results
