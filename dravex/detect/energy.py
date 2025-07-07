"""Energy and loudness detection."""
import math
from dravex.constants import power_to_db, DEFAULT_EPSILON

def frame_energy(x, frame_size=1024, hop_size=512):
    """Compute mean energy per frame."""
    """Compute energy per frame."""
    n = len(x)
    energies = []
    pos = 0
    while pos + frame_size <= n:
        e = sum(s * s for s in x[pos:pos + frame_size]) / frame_size
        energies.append(e)
        pos += hop_size
    return energies

def frame_rms(x, frame_size=1024, hop_size=512):
    """Compute RMS per frame."""
    return [math.sqrt(e) for e in frame_energy(x, frame_size, hop_size)]

def frame_db(x, frame_size=1024, hop_size=512):
    """Frame-level energy in dB."""
    return [power_to_db(e) for e in frame_energy(x, frame_size, hop_size)]

def loudness_curve(x, frame_size=2048, hop_size=512):
    """Approximate loudness (A-weighted energy)."""
    from dravex.transform.fft import rfft
    n = len(x)
    loudness = []
    pos = 0
    while pos + frame_size <= n:
        frame = x[pos:pos + frame_size]
        mags = rfft(frame)
        weighted_power = 0.0
        for k, m in enumerate(mags):
            freq = k * 44100 / frame_size
            w = _a_weight(freq)
            weighted_power += (m * w) ** 2
        loudness.append(weighted_power / len(mags))
        pos += hop_size
    return loudness

def _a_weight(freq):
    """Simplified A-weighting factor."""
    if freq < 20 or freq > 20000:
        return 0.0
    f2 = freq * freq
    num = 12194**2 * f2 * f2
    denom = (f2 + 20.6**2) * math.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
    if denom < DEFAULT_EPSILON:
        return 0.0
    return num / denom

def total_energy(x):
    """Total signal energy."""
    return sum(s * s for s in x)

def short_time_energy(x, frame_size=256, hop_size=128):
    """Short-time energy curve."""
    n = len(x)
    ste = []
    pos = 0
    while pos + frame_size <= n:
        e = sum(s * s for s in x[pos:pos + frame_size])
        ste.append(e)
        pos += hop_size
    return ste

def energy_ratio(x, low_freq, high_freq, sample_rate, frame_size=2048):
    """Ratio of energy in low band to high band."""
    from dravex.transform.fft import rfft
    mags = rfft(x[:frame_size])
    n_fft = frame_size
    low_bin = int(low_freq * n_fft / sample_rate)
    high_bin = int(high_freq * n_fft / sample_rate)
    low_energy = sum(mags[i]**2 for i in range(min(low_bin, len(mags))))
    high_energy = sum(mags[i]**2 for i in range(min(high_bin, len(mags)), len(mags)))
    if high_energy < 1e-12:
        return float('inf')
    return low_energy / high_energy

def spectral_centroid_envelope(x, frame_size=2048, hop_size=512, sample_rate=44100):
    """Track spectral centroid over time."""
    from dravex.transform.fft import rfft
    n = len(x)
    centroids = []
    pos = 0
    while pos + frame_size <= n:
        frame = x[pos:pos + frame_size]
        mags = rfft(frame)
        total = sum(mags)
        if total > 1e-12:
            weighted = sum(i * m for i, m in enumerate(mags))
            c = (weighted / total) * sample_rate / frame_size
        else:
            c = 0.0
        centroids.append(c)
        pos += hop_size
    return centroids
