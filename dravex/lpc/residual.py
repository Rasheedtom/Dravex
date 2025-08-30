"""LPC residual analysis and manipulation."""
import math
from dravex.constants import DEFAULT_EPSILON

def extract_residual(x, order=12):
    """Extract LPC residual (excitation signal)."""
    from dravex.lpc.analysis import lpc_autocorrelation, lpc_error
    coeffs = lpc_autocorrelation(x, order)
    return lpc_error(x, coeffs)

def residual_energy(residual):
    """Energy of LPC residual."""
    return sum(r * r for r in residual)

def prediction_gain(x, residual):
    """LPC prediction gain in dB."""
    sig_energy = sum(s * s for s in x)
    res_energy = sum(r * r for r in residual)
    if res_energy < DEFAULT_EPSILON:
        return 100.0
    return 10 * math.log10(sig_energy / res_energy)

def classify_excitation(residual, threshold=3.0):
    """Classify residual as voiced (pulse) or unvoiced (noise)."""
    if not residual:
        return "unvoiced"
    peak = max(abs(r) for r in residual)
    rms_val = math.sqrt(sum(r * r for r in residual) / len(residual))
    if rms_val < DEFAULT_EPSILON:
        return "unvoiced"
    crest = peak / rms_val
    return "voiced" if crest > threshold else "unvoiced"

def whiten(x, order=12):
    """Spectral whitening via inverse filtering (extract residual)."""
    return extract_residual(x, order)

def residual_spectrum(residual, n_fft=512):
    """Compute spectrum of LPC residual."""
    from dravex.transform.fft import rfft
    padded = list(residual[:n_fft]) + [0.0] * max(0, n_fft - len(residual))
    return rfft(padded)

def pitch_from_residual(residual, sample_rate=8000, min_freq=50, max_freq=400):
    """Detect pitch from autocorrelation of LPC residual."""
    n = len(residual)
    min_lag = max(1, int(sample_rate / max_freq))
    max_lag = min(n // 2, int(sample_rate / min_freq))
    best_lag = 0
    best_val = 0.0
    for lag in range(min_lag, max_lag):
        val = sum(residual[i] * residual[i + lag] for i in range(n - lag))
        if val > best_val:
            best_val = val
            best_lag = lag
    return sample_rate / best_lag if best_lag > 0 else 0.0

def spectral_tilt(residual, n_fft=512):
    """Estimate spectral tilt from residual."""
    import math
    mags = residual_spectrum(residual, n_fft)
    if len(mags) < 2:
        return 0.0
    log_mags = [math.log(max(m, 1e-12)) for m in mags]
    n = len(log_mags)
    x_mean = (n - 1) / 2
    y_mean = sum(log_mags) / n
    num = sum((i - x_mean) * (log_mags[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if abs(den) > 1e-12 else 0.0
