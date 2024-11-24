"""Tests for STFT."""
from dravex.transform.stft import stft, istft, spectrogram
import math
from dravex.transform.stft import stft, istft, spectrogram, griffin_lim, phase_vocoder, mel_spectrogram

class TestSTFT:
    def test_stft_basic(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        frames = stft(x, 1024, 256)
        assert len(frames) > 0
        assert len(frames[0][0]) > 0

    def test_spectrogram(self):
        x = [0.5] * 4096
        spec = spectrogram(x, 1024, 512)
        assert len(spec) > 0

    def test_roundtrip(self):
        x = list(range(2048))
        x = [float(v)/2048 for v in x]
        frames = stft(x, 256, 64)
        y = istft(frames, 256, 64)
        for i in range(100, len(x) - 100):
            assert abs(x[i] - y[i]) < 0.1

    def test_frame_count(self):
        x = [0.0] * 2048
        frames = stft(x, 512, 256)
        expected = (2048 - 512) // 256 + 1
        assert len(frames) == expected



class TestSTFTAdvanced:
    def test_griffin_lim(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        spec = spectrogram(x, 512, 128)
        reconstructed = griffin_lim(spec, 10, 512, 128)
        assert len(reconstructed) > 0

    def test_phase_vocoder(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        frames = stft(x, 512, 128)
        stretched = phase_vocoder(frames, 1.5, 128)
        assert len(stretched) > len(frames)

    def test_mel_spectrogram(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        mel = mel_spectrogram(x, 44100, 512, 128, 40)
        assert len(mel) > 0
        assert len(mel[0]) == 40

    def test_roundtrip_sine(self):
        freq = 440
        sr = 44100
        x = [math.sin(2*math.pi*freq*i/sr) for i in range(4096)]
        frames = stft(x, 1024, 256)
        y = istft(frames, 1024, 256)
        for i in range(500, 3500):
            assert abs(x[i] - y[i]) < 0.2

    def test_spectrogram_shape(self):
        x = [0.0] * 4096
        spec = spectrogram(x, 512, 256)
        expected_frames = (4096 - 512) // 256 + 1
        assert len(spec) == expected_frames
        assert len(spec[0]) == 257

    def test_time_stretch(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(8192)]
        frames = stft(x, 1024, 256)
        slow = phase_vocoder(frames, 2.0, 256)
        y = istft(slow, 1024, 256)
        assert len(y) > len(x) * 1.5
