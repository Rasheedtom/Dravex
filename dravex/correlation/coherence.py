"""Spectral coherence between signals."""
import math
from dravex.transform.stft import stft
from dravex.constants import DEFAULT_EPSILON

def coherence(x, y, frame_size=1024, hop_size=512, sample_rate=44100):
    """Magnitude-squared coherence between two signals."""
    from dravex.window.hann import hann
    window = hann(frame_size)
    frames_x = stft(x, frame_size, hop_size, window)
    frames_y = stft(y, frame_size, hop_size, window)
    n_frames = min(len(frames_x), len(frames_y))
    n_bins = len(frames_x[0][0]) if frames_x else 0
    Sxx = [0.0] * n_bins
    Syy = [0.0] * n_bins
    Sxy_re = [0.0] * n_bins
    Sxy_im = [0.0] * n_bins
    for f in range(n_frames):
        mx, px = frames_x[f]
        my, py = frames_y[f]
        for k in range(n_bins):
            xr = mx[k] * math.cos(px[k])
            xi = mx[k] * math.sin(px[k])
            yr = my[k] * math.cos(py[k])
            yi = my[k] * math.sin(py[k])
            Sxx[k] += mx[k] * mx[k]
            Syy[k] += my[k] * my[k]
            Sxy_re[k] += xr * yr + xi * yi
            Sxy_im[k] += xi * yr - xr * yi
    coh = []
    for k in range(n_bins):
        cross_mag_sq = Sxy_re[k]**2 + Sxy_im[k]**2
        denom = Sxx[k] * Syy[k]
        if denom > DEFAULT_EPSILON:
            coh.append(cross_mag_sq / denom)
        else:
            coh.append(0.0)
    freqs = [k * sample_rate / frame_size for k in range(n_bins)]
    return freqs, coh

def transfer_function(x, y, frame_size=1024, hop_size=512, sample_rate=44100):
    """Estimate transfer function H(f) = Sxy/Sxx."""
    import math
    from dravex.transform.stft import stft
    from dravex.window.hann import hann
    window = hann(frame_size)
    frames_x = stft(x, frame_size, hop_size, window)
    frames_y = stft(y, frame_size, hop_size, window)
    n_frames = min(len(frames_x), len(frames_y))
    n_bins = len(frames_x[0][0]) if frames_x else 0
    Sxx = [0.0] * n_bins
    Sxy_re = [0.0] * n_bins
    Sxy_im = [0.0] * n_bins
    for f in range(n_frames):
        mx, px = frames_x[f]
        my, py = frames_y[f]
        for k in range(n_bins):
            xr = mx[k] * math.cos(px[k])
            xi = mx[k] * math.sin(px[k])
            yr = my[k] * math.cos(py[k])
            yi = my[k] * math.sin(py[k])
            Sxx[k] += xr*xr + xi*xi
            Sxy_re[k] += xr*yr + xi*yi
            Sxy_im[k] += xi*yr - xr*yi
    H_mag = []
    H_phase = []
    for k in range(n_bins):
        if Sxx[k] > 1e-12:
            mag = math.sqrt(Sxy_re[k]**2 + Sxy_im[k]**2) / Sxx[k]
            phase = math.atan2(Sxy_im[k], Sxy_re[k])
        else:
            mag = 0.0
            phase = 0.0
        H_mag.append(mag)
        H_phase.append(phase)
    freqs = [k * sample_rate / frame_size for k in range(n_bins)]
    return freqs, H_mag, H_phase
