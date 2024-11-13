"""Short-Time Fourier Transform."""
import math
from dravex.transform.fft import fft, rfft, ifft
from dravex.constants import next_power_of_2

def stft(signal_samples, frame_size=1024, hop_size=256, window=None):
    """Compute STFT returning list of (magnitudes, phases) per frame."""
    n = len(signal_samples)
    if window is None:
        window = _hann(frame_size)
    frames = []
    pos = 0
    while pos + frame_size <= n:
        chunk = signal_samples[pos:pos + frame_size]
        windowed = [s * w for s, w in zip(chunk, window)]
        N = next_power_of_2(len(windowed))
        padded = windowed + [0.0] * (N - len(windowed))
        X = fft(padded)
        n_bins = N // 2 + 1
        mags = [math.sqrt(X[k][0]**2 + X[k][1]**2) for k in range(n_bins)]
        phases = [math.atan2(X[k][1], X[k][0]) for k in range(n_bins)]
        frames.append((mags, phases))
        pos += hop_size
    return frames

def istft(stft_frames, frame_size=1024, hop_size=256, window=None):
    """Inverse STFT using overlap-add."""
    if not stft_frames:
        return []
    n_bins = len(stft_frames[0][0])
    N = (n_bins - 1) * 2
    if window is None:
        window = _hann(frame_size)
    total_length = frame_size + hop_size * (len(stft_frames) - 1)
    output = [0.0] * total_length
    window_sum = [0.0] * total_length
    for i, (mags, phases) in enumerate(stft_frames):
        X = []
        for k in range(N):
            if k < n_bins:
                re = mags[k] * math.cos(phases[k])
                im = mags[k] * math.sin(phases[k])
            else:
                mirror = N - k
                re = mags[mirror] * math.cos(-phases[mirror])
                im = mags[mirror] * math.sin(-phases[mirror])
            X.append((re, im))
        x = ifft(X)
        pos = i * hop_size
        for j in range(min(frame_size, N)):
            if pos + j < total_length:
                output[pos + j] += x[j][0] * window[j] if j < len(window) else x[j][0]
                window_sum[pos + j] += window[j] ** 2 if j < len(window) else 1.0
    for i in range(total_length):
        if window_sum[i] > 1e-8:
            output[i] /= window_sum[i]
    return output[:total_length]

def _hann(n):
    return [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]

def spectrogram(signal_samples, frame_size=1024, hop_size=256, window=None):
    """Compute spectrogram (magnitude STFT)."""
    frames = stft(signal_samples, frame_size, hop_size, window)
    return [mags for mags, _ in frames]

def griffin_lim(magnitudes, n_iter=30, frame_size=1024, hop_size=256):
    """Griffin-Lim phase reconstruction from magnitude spectrogram."""
    import math
    from dravex.constants import TAU
    n_frames = len(magnitudes)
    n_bins = len(magnitudes[0])
    N = (n_bins - 1) * 2
    phases = [[0.0] * n_bins for _ in range(n_frames)]
    for iteration in range(n_iter):
        frames_with_phase = [(magnitudes[i], phases[i]) for i in range(n_frames)]
        signal = istft(frames_with_phase, frame_size, hop_size)
        new_frames = stft(signal, frame_size, hop_size)
        for i in range(min(len(new_frames), n_frames)):
            _, new_phase = new_frames[i]
            phases[i] = new_phase
    final_frames = [(magnitudes[i], phases[i]) for i in range(n_frames)]
    return istft(final_frames, frame_size, hop_size)

def phase_vocoder(stft_frames, stretch_factor, hop_size):
    """Time stretching via phase vocoder."""
    import math
    n_frames = len(stft_frames)
    n_bins = len(stft_frames[0][0])
    new_n_frames = int(n_frames * stretch_factor)
    result = []
    for i in range(new_n_frames):
        src_pos = i / stretch_factor
        idx = int(src_pos)
        frac = src_pos - idx
        if idx + 1 < n_frames:
            mags_a, phases_a = stft_frames[idx]
            mags_b, phases_b = stft_frames[idx + 1]
            new_mags = [ma * (1 - frac) + mb * frac for ma, mb in zip(mags_a, mags_b)]
            new_phases = []
            for k in range(n_bins):
                dp = phases_b[k] - phases_a[k]
                expected = 2 * math.pi * k * hop_size / ((n_bins - 1) * 2)
                dp_wrapped = dp - expected
                while dp_wrapped > math.pi: dp_wrapped -= 2 * math.pi
                while dp_wrapped < -math.pi: dp_wrapped += 2 * math.pi
                new_phases.append(phases_a[k] + (expected + dp_wrapped) * (1 + frac * (stretch_factor - 1)))
            result.append((new_mags, new_phases))
        elif idx < n_frames:
            result.append(stft_frames[idx])
    return result

def mel_spectrogram(signal_samples, sample_rate=44100, frame_size=1024, hop_size=256, n_mels=128):
    """Compute mel spectrogram."""
    from dravex.spectral.mfcc import mel_filterbank
    spec = spectrogram(signal_samples, frame_size, hop_size)
    mel_fb = mel_filterbank(n_mels, frame_size, sample_rate)
    mel_spec = []
    for frame in spec:
        mel_frame = []
        for filt in mel_fb:
            energy = sum(m * f for m, f in zip(frame, filt))
            mel_frame.append(max(energy, 1e-10))
        mel_spec.append(mel_frame)
    return mel_spec
