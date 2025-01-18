"""Phase spectrum analysis and unwrapping."""
import math
from dravex.transform.fft import fft
from dravex.constants import PI, TAU, DEFAULT_EPSILON

def phase_spectrum(x):
    """Compute phase spectrum."""
    X = fft(x)
    n = len(X) // 2 + 1
    return [math.atan2(X[i][1], X[i][0]) for i in range(n)]

def unwrap_phase(phases):
    """Phase unwrapping: remove 2*pi discontinuities."""
    result = [phases[0]]
    for i in range(1, len(phases)):
        diff = phases[i] - phases[i-1]
        while diff > PI:
            diff -= TAU
        while diff < -PI:
            diff += TAU
        result.append(result[-1] + diff)
    return result

def group_delay_from_phase(phases, sample_rate, n_fft):
    """Compute group delay from phase spectrum."""
    unwrapped = unwrap_phase(phases)
    gd = [0.0]
    freq_step = sample_rate / n_fft
    for i in range(1, len(unwrapped)):
        gd.append(-(unwrapped[i] - unwrapped[i-1]) / (TAU * freq_step))
    return gd

def instantaneous_frequency_from_phase(phases, hop_size, sample_rate):
    """Instantaneous frequency from STFT phase differences."""
    freqs = []
    for i in range(len(phases)):
        frame_freqs = []
        for k in range(len(phases[i])):
            if i == 0:
                frame_freqs.append(0.0)
            else:
                dp = phases[i][k] - phases[i-1][k] - TAU * k * hop_size / len(phases[i]) / 2
                while dp > PI: dp -= TAU
                while dp < -PI: dp += TAU
                freq = (TAU * k / len(phases[i]) / 2 + dp / hop_size) * sample_rate / TAU
                frame_freqs.append(freq)
        freqs.append(frame_freqs)
    return freqs

def minimum_phase(magnitude_spectrum):
    """Compute minimum phase from magnitude spectrum."""
    log_mag = [math.log(max(m, DEFAULT_EPSILON)) for m in magnitude_spectrum]
    from dravex.transform.fft import fft, ifft
    n = len(log_mag)
    symmetric = log_mag + list(reversed(log_mag[1:-1] if n > 1 else []))
    cepstrum = ifft([(v, 0.0) for v in symmetric])
    n_full = len(cepstrum)
    causal = [(0.0, 0.0)] * n_full
    causal[0] = cepstrum[0]
    for i in range(1, n_full // 2):
        causal[i] = (2 * cepstrum[i][0], 2 * cepstrum[i][1])
    if n_full > 1:
        causal[n_full // 2] = cepstrum[n_full // 2]
    result = fft([r for r, _ in causal])
    return [math.atan2(im, re) for re, im in result[:n]]

def phase_difference(phases1, phases2):
    """Compute phase difference between two spectra."""
    import math
    result = []
    for p1, p2 in zip(phases1, phases2):
        diff = p2 - p1
        while diff > math.pi: diff -= 2*math.pi
        while diff < -math.pi: diff += 2*math.pi
        result.append(diff)
    return result

def coherent_phase(magnitudes, phases):
    """Check phase coherence across frequency bins."""
    import math
    n = len(phases)
    if n < 3:
        return 1.0
    diffs = []
    for i in range(1, n):
        d = phases[i] - phases[i-1]
        while d > math.pi: d -= 2*math.pi
        while d < -math.pi: d += 2*math.pi
        diffs.append(d)
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff)**2 for d in diffs) / len(diffs)
    return math.exp(-variance)

def phase_vocoder_stretch(phases_prev, phases_curr, hop_size, n_fft, stretch_factor):
    """Compute stretched phase for phase vocoder."""
    import math
    n_bins = len(phases_curr)
    new_phases = []
    for k in range(n_bins):
        expected = 2 * math.pi * k * hop_size / n_fft
        dp = phases_curr[k] - phases_prev[k] - expected
        while dp > math.pi: dp -= 2*math.pi
        while dp < -math.pi: dp += 2*math.pi
        true_freq = expected + dp
        new_phases.append(phases_prev[k] + true_freq * stretch_factor)
    return new_phases
