"""Harmonic analysis and extraction."""
import math
from dravex.transform.fft import rfft
from dravex.constants import DEFAULT_EPSILON

def find_harmonics(magnitudes, fundamental_bin, n_harmonics=8, tolerance=2):
    """Find harmonic series from a magnitude spectrum."""
    harmonics = []
    for h in range(1, n_harmonics + 1):
        target = fundamental_bin * h
        best_bin = target
        best_mag = 0.0
        for offset in range(-tolerance, tolerance + 1):
            idx = target + offset
            if 0 <= idx < len(magnitudes) and magnitudes[idx] > best_mag:
                best_mag = magnitudes[idx]
                best_bin = idx
        harmonics.append((best_bin, best_mag))
    return harmonics

def harmonic_to_noise_ratio(x, sample_rate, fundamental_freq):
    """Estimate harmonic-to-noise ratio."""
    mags = rfft(x)
    n_fft = (len(mags) - 1) * 2
    fund_bin = int(round(fundamental_freq * n_fft / sample_rate))
    if fund_bin <= 0:
        return 0.0
    harmonics = find_harmonics(mags, fund_bin)
    harm_power = sum(m * m for _, m in harmonics)
    total_power = sum(m * m for m in mags)
    noise_power = total_power - harm_power
    if noise_power < DEFAULT_EPSILON:
        return 100.0
    return 10 * math.log10(harm_power / noise_power)

def inharmonicity(harmonics, fundamental_bin):
    """Measure deviation of harmonics from ideal integer ratios."""
    if not harmonics or fundamental_bin == 0:
        return 0.0
    total_dev = 0.0
    for i, (bin_idx, mag) in enumerate(harmonics):
        expected = fundamental_bin * (i + 1)
        deviation = abs(bin_idx - expected) / max(1, expected)
        total_dev += deviation * mag
    total_mag = sum(m for _, m in harmonics)
    return total_dev / total_mag if total_mag > DEFAULT_EPSILON else 0.0

def odd_even_ratio(harmonics):
    """Ratio of odd to even harmonic energy."""
    odd_power = sum(m*m for i, (_, m) in enumerate(harmonics) if (i+1) % 2 == 1)
    even_power = sum(m*m for i, (_, m) in enumerate(harmonics) if (i+1) % 2 == 0)
    if even_power < DEFAULT_EPSILON:
        return float('inf')
    return odd_power / even_power

def spectral_flatness_ratio(magnitudes, fundamental_bin, n_harmonics=8):
    """Ratio of spectral flatness at harmonics vs non-harmonics."""
    import math
    harmonic_bins = set()
    for h in range(1, n_harmonics + 1):
        target = fundamental_bin * h
        for offset in range(-2, 3):
            idx = target + offset
            if 0 <= idx < len(magnitudes):
                harmonic_bins.add(idx)
    harmonic_mags = [magnitudes[i] for i in range(len(magnitudes)) if i in harmonic_bins and magnitudes[i] > 1e-12]
    non_harmonic_mags = [magnitudes[i] for i in range(len(magnitudes)) if i not in harmonic_bins and magnitudes[i] > 1e-12]
    def flatness(mags):
        if not mags:
            return 0.0
        log_mean = sum(math.log(m) for m in mags) / len(mags)
        arith_mean = sum(mags) / len(mags)
        return math.exp(log_mean) / arith_mean if arith_mean > 0 else 0
    hf = flatness(harmonic_mags)
    nf = flatness(non_harmonic_mags)
    return hf / nf if nf > 1e-12 else 0.0

def tristimulus(harmonics):
    """Tristimulus values (timbre descriptors from harmonics)."""
    if len(harmonics) < 4:
        return (0.0, 0.0, 0.0)
    total = sum(m for _, m in harmonics)
    if total < 1e-12:
        return (0.0, 0.0, 0.0)
    t1 = harmonics[0][1] / total
    t2 = sum(m for _, m in harmonics[1:4]) / total if len(harmonics) >= 4 else 0
    t3 = sum(m for _, m in harmonics[4:]) / total if len(harmonics) > 4 else 0
    return (t1, t2, t3)
