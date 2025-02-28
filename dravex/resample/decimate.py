"""Signal decimation (downsampling)."""
from dravex.filter.fir import fir_lowpass, apply_fir

def decimate(x, factor, sample_rate=44100):
    """Downsample by integer factor with anti-alias filtering.

    Applies a lowpass FIR filter at the new Nyquist frequency
    before downsampling to prevent aliasing.
    """
    """Downsample by integer factor with anti-alias filtering."""
    nyquist = sample_rate / (2 * factor)
    h = fir_lowpass(min(64, 8 * factor), nyquist * 0.9, sample_rate)
    filtered = apply_fir(x, h)
    return filtered[::factor]

def decimate_no_filter(x, factor):
    """Simple decimation without anti-alias filter."""
    return x[::factor]

def downsample_average(x, factor):
    """Downsample by averaging groups of samples."""
    n = len(x)
    result = []
    for i in range(0, n - factor + 1, factor):
        avg = sum(x[i:i + factor]) / factor
        result.append(avg)
    return result

def downsample_peak(x, factor):
    """Downsample by taking peak of each group."""
    n = len(x)
    result = []
    for i in range(0, n - factor + 1, factor):
        peak = max(abs(s) for s in x[i:i + factor])
        result.append(peak)
    return result

def multi_rate(x, up_factor, down_factor, sample_rate=44100):
    """Rational resampling: upsample then downsample."""
    from dravex.resample.interpolate import linear_interpolate
    upsampled = linear_interpolate(x, up_factor)
    return decimate(upsampled, down_factor, sample_rate * up_factor)

def anti_alias_downsample(x, factor, filter_order=64, sample_rate=44100):
    """Downsample with proper anti-aliasing."""
    nyquist = sample_rate / (2 * factor)
    cutoff = nyquist * 0.85
    from dravex.filter.fir import fir_lowpass, apply_fir
    h = fir_lowpass(filter_order, cutoff, sample_rate)
    filtered = apply_fir(x, h)
    return filtered[::factor]

def polyphase_decimate(x, factor, filter_order=32, sample_rate=44100):
    """Polyphase decimation (compute only needed output samples)."""
    from dravex.filter.fir import fir_lowpass
    nyquist = sample_rate / (2 * factor)
    h = fir_lowpass(filter_order, nyquist * 0.9, sample_rate)
    n_out = len(x) // factor
    result = [0.0] * n_out
    for i in range(n_out):
        s = 0.0
        for j in range(len(h)):
            idx = i * factor - j
            if 0 <= idx < len(x):
                s += h[j] * x[idx]
        result[i] = s
    return result

def cascade_decimate(x, total_factor, sample_rate=44100):
    """Multi-stage decimation for large factors."""
    result = list(x)
    sr = sample_rate
    remaining = total_factor
    while remaining > 1:
        stage_factor = min(remaining, 4)
        result = anti_alias_downsample(result, stage_factor, 32, sr)
        sr //= stage_factor
        remaining //= stage_factor
    return result
