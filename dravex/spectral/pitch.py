"""Pitch detection algorithms."""
import math
from dravex.constants import DEFAULT_EPSILON
from dravex.config import Config

def autocorrelation_pitch(x, sample_rate=None):
    """Pitch detection via autocorrelation method."""
    cfg = Config.get()
    if sample_rate is None:
        sample_rate = cfg.default_sample_rate
    min_period = int(sample_rate / cfg.pitch_max_hz)
    max_period = min(int(sample_rate / cfg.pitch_min_hz), len(x) // 2)
    if max_period <= min_period:
        return 0.0
    best_corr = 0.0
    best_lag = 0
    n = len(x)
    for lag in range(min_period, max_period):
        corr = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for i in range(n - lag):
            corr += x[i] * x[i + lag]
            norm1 += x[i] * x[i]
            norm2 += x[i + lag] * x[i + lag]
        denom = math.sqrt(norm1 * norm2)
        if denom > DEFAULT_EPSILON:
            corr /= denom
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_lag == 0 or best_corr < 0.3:
        return 0.0
    return sample_rate / best_lag

def yin_pitch(x, sample_rate=None, threshold=0.1):
    """YIN pitch detection algorithm.

    The threshold parameter controls sensitivity (lower = stricter).
    """
    """YIN pitch detection algorithm."""
    cfg = Config.get()
    if sample_rate is None:
        sample_rate = cfg.default_sample_rate
    n = len(x)
    max_tau = min(n // 2, int(sample_rate / cfg.pitch_min_hz))
    min_tau = max(2, int(sample_rate / cfg.pitch_max_hz))
    d = [0.0] * max_tau
    for tau in range(1, max_tau):
        for i in range(n - max_tau):
            diff = x[i] - x[i + tau]
            d[tau] += diff * diff
    d_prime = [1.0] * max_tau
    cumulative = 0.0
    for tau in range(1, max_tau):
        cumulative += d[tau]
        if cumulative > DEFAULT_EPSILON:
            d_prime[tau] = d[tau] * tau / cumulative
    for tau in range(min_tau, max_tau - 1):
        if d_prime[tau] < threshold:
            if d_prime[tau] <= d_prime[tau + 1]:
                return sample_rate / tau
    return 0.0

def cepstral_pitch(x, sample_rate=None):
    """Pitch detection via cepstral analysis."""
    from dravex.transform.cepstrum import real_cepstrum, cepstral_peak
    cfg = Config.get()
    if sample_rate is None:
        sample_rate = cfg.default_sample_rate
    cep = real_cepstrum(x)
    return cepstral_peak(cep, sample_rate, cfg.pitch_min_hz, cfg.pitch_max_hz)

def multi_pitch(x, sample_rate=None, max_voices=3):
    """Detect multiple simultaneous pitches."""
    from dravex.transform.fft import rfft
    from dravex.detect.peaks import find_peaks
    cfg = Config.get()
    if sample_rate is None:
        sample_rate = cfg.default_sample_rate
    mags = rfft(x)
    peaks = find_peaks(mags, min_distance=5, min_height=max(mags) * 0.1)
    pitches = []
    for p in peaks[:max_voices * 3]:
        freq = p * sample_rate / (2 * (len(mags) - 1))
        if cfg.pitch_min_hz <= freq <= cfg.pitch_max_hz:
            pitches.append(freq)
    return sorted(set(pitches))[:max_voices]

def pitch_contour(x, sample_rate=None, frame_size=2048, hop_size=512):
    """Extract pitch contour over time."""
    if sample_rate is None:
        sample_rate = Config.get().default_sample_rate
    n = len(x)
    contour = []
    pos = 0
    while pos + frame_size <= n:
        frame = x[pos:pos + frame_size]
        p = autocorrelation_pitch(frame, sample_rate)
        contour.append(p)
        pos += hop_size
    return contour

def pitch_strength_metric(x, sample_rate=None):
    """Measure how strongly periodic (pitched) the signal is."""
    import math
    if sample_rate is None:
        sample_rate = Config.get().default_sample_rate
    n = len(x)
    if n < 100:
        return 0.0
    r = [0.0] * (n // 2)
    for lag in range(len(r)):
        for i in range(n - lag):
            r[lag] += x[i] * x[i + lag]
    if r[0] < 1e-12:
        return 0.0
    normalized = [ri / r[0] for ri in r]
    min_lag = max(2, int(sample_rate / Config.get().pitch_max_hz))
    max_lag = min(len(r), int(sample_rate / Config.get().pitch_min_hz))
    if min_lag >= max_lag:
        return 0.0
    peak = max(normalized[min_lag:max_lag])
    return max(0, peak)
