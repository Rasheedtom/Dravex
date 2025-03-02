"""Polyphase resampling for efficient multi-rate conversion."""
from dravex.filter.fir import fir_lowpass

def polyphase_upsample(x, factor, filter_length=32, sample_rate=44100):
    """Polyphase upsampling: split filter into sub-filters."""
    cutoff = sample_rate / (2 * factor) * 0.9
    h = fir_lowpass(filter_length * factor, cutoff, sample_rate * factor)
    sub_filters = _decompose(h, factor)
    out_len = len(x) * factor
    y = [0.0] * out_len
    for phase, sub_h in enumerate(sub_filters):
        for i in range(len(x)):
            for j in range(len(sub_h)):
                idx = i - j
                if 0 <= idx < len(x):
                    y[i * factor + phase] += sub_h[j] * x[idx]
    return y

def polyphase_downsample(x, factor, filter_length=32, sample_rate=44100):
    """Polyphase downsampling."""
    cutoff = sample_rate / (2 * factor) * 0.9
    h = fir_lowpass(filter_length, cutoff, sample_rate)
    n_out = len(x) // factor
    y = [0.0] * n_out
    for i in range(n_out):
        s = 0.0
        for j in range(len(h)):
            idx = i * factor - j
            if 0 <= idx < len(x):
                s += h[j] * x[idx]
        y[i] = s
    return y

def _decompose(h, n_phases):
    """Decompose filter into polyphase components."""
    sub_filters = [[] for _ in range(n_phases)]
    for i, coeff in enumerate(h):
        sub_filters[i % n_phases].append(coeff)
    return sub_filters

def polyphase_resample(x, up, down, sample_rate=44100):
    """Rational resampling using polyphase structure."""
    upsampled = polyphase_upsample(x, up, sample_rate=sample_rate)
    return polyphase_downsample(upsampled, down, sample_rate=sample_rate * up)

def polyphase_analysis(x, n_channels):
    """Polyphase analysis filterbank (split into sub-bands)."""
    channels = [[] for _ in range(n_channels)]
    for i, s in enumerate(x):
        channels[i % n_channels].append(s)
    return channels

def polyphase_synthesis(channels):
    """Polyphase synthesis filterbank (combine sub-bands)."""
    n_channels = len(channels)
    max_len = max(len(ch) for ch in channels) if channels else 0
    result = []
    for i in range(max_len):
        for ch in range(n_channels):
            if i < len(channels[ch]):
                result.append(channels[ch][i])
    return result

def channelizer(x, n_channels, filter_order=32, sample_rate=44100):
    """Split signal into frequency channels using polyphase."""
    from dravex.filter.fir import fir_lowpass, apply_fir
    band_width = sample_rate / (2 * n_channels)
    channels = []
    for ch in range(n_channels):
        low = ch * band_width
        high = (ch + 1) * band_width
        if ch == 0:
            h = fir_lowpass(filter_order, high, sample_rate)
        else:
            from dravex.filter.fir import fir_bandpass
            h = fir_bandpass(filter_order, max(1, low), min(high, sample_rate/2 - 1), sample_rate)
        filtered = apply_fir(x, h)
        channels.append(filtered)
    return channels
