"""Tests for signal I/O."""
import os, tempfile
from dravex.io.wav import write_wav, read_wav
from dravex.io.raw import write_raw_f32, read_raw_f32
from dravex.io.csv_io import write_csv, read_csv
from dravex.io.tone_gen import sine_wave, square_wave, impulse
from dravex.io.json_signal import to_json, from_json
import os, tempfile, math
from dravex.io.raw import write_raw_f32, read_raw_f32, write_raw_i16, read_raw_i16, samples_to_bytes
from dravex.io.tone_gen import sine_wave, sawtooth_wave, triangle_wave, sweep, multi_tone, dtmf_tone
from dravex.io.spectrogram_io import spectrogram_to_ascii, save_spectrogram_csv
from dravex.io.json_signal import to_json, from_json, save_json, load_json, multi_signal_json

class TestIO:
    def test_wav_roundtrip(self):
        samples = [0.5, -0.3, 0.8, -0.1]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        write_wav(path, samples, 44100, 16)
        read_samples, sr = read_wav(path)
        os.unlink(path)
        assert sr == 44100
        for a, b in zip(samples, read_samples):
            assert abs(a - b) < 0.001

    def test_raw_roundtrip(self):
        samples = [1.5, -2.3, 0.0]
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
            path = f.name
        write_raw_f32(path, samples)
        read_back = read_raw_f32(path)
        os.unlink(path)
        for a, b in zip(samples, read_back):
            assert abs(a - b) < 1e-6

    def test_csv_roundtrip(self):
        signals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            path = f.name
        write_csv(path, signals, headers=['a', 'b'])
        read_back = read_csv(path)
        os.unlink(path)
        assert len(read_back) == 2

    def test_sine_wave(self):
        s = sine_wave(440, 0.01, 44100)
        assert len(s) == 441

    def test_square_wave(self):
        s = square_wave(1000, 0.001, 44100)
        assert len(s) > 0

    def test_impulse(self):
        s = impulse(10, 3)
        assert s[3] == 1.0
        assert s[0] == 0.0

    def test_json_roundtrip(self):
        samples = [1.0, 2.0, 3.0]
        j = to_json(samples, 44100)
        s, sr, _ = from_json(j)
        assert s == samples



class TestIOAdvanced:
    def test_wav_8bit(self):
        samples = [0.5, -0.3]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        write_wav(path, samples, 44100, 8)
        read_back, sr = read_wav(path)
        os.unlink(path)
        assert sr == 44100

    def test_raw_i16_roundtrip(self):
        samples = [0.5, -0.3, 0.0]
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
            path = f.name
        write_raw_i16(path, samples)
        read_back = read_raw_i16(path)
        os.unlink(path)
        for a, b in zip(samples, read_back):
            assert abs(a - b) < 0.001

    def test_samples_to_bytes(self):
        data = samples_to_bytes([0.5, -0.5], 16)
        assert len(data) == 4

    def test_sawtooth(self):
        s = sawtooth_wave(440, 0.01, 44100)
        assert len(s) > 0

    def test_triangle(self):
        s = triangle_wave(440, 0.01, 44100)
        assert len(s) > 0

    def test_sweep(self):
        s = sweep(100, 5000, 0.1, 44100)
        assert len(s) == 4410

    def test_multi_tone(self):
        s = multi_tone([440, 880, 1320], 0.01, 44100)
        assert len(s) > 0

    def test_dtmf(self):
        s = dtmf_tone('5', 0.05, 44100)
        assert len(s) > 0

    def test_spectrogram_ascii(self):
        spec = [[float(i+j) for i in range(10)] for j in range(20)]
        ascii_art = spectrogram_to_ascii(spec, 5, 10)
        assert len(ascii_art) > 0

    def test_json_roundtrip(self):
        samples = [0.1, 0.2, 0.3]
        j = to_json(samples, 44100, {"key": "value"})
        s, sr, meta = from_json(j)
        assert sr == 44100
        assert meta["key"] == "value"

    def test_json_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        save_json(path, [1.0, 2.0], 44100)
        s, sr, _ = load_json(path)
        os.unlink(path)
        assert s == [1.0, 2.0]

    def test_multi_signal_json(self):
        j = multi_signal_json([[1, 2], [3, 4]], ["a", "b"])
        import json
        obj = json.loads(j)
        assert "a" in obj["signals"]

    def test_csv_with_time(self):
        signals = [[1.0, 2.0]]
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            path = f.name
        write_csv(path, signals, sample_rate=44100)
        os.unlink(path)


class TestToneGenExtra:
    def test_am_tone(self):
        from dravex.io.tone_gen import am_tone
        t = am_tone(1000, 5, 0.01, 44100)
        assert len(t) == 441

    def test_fm_tone(self):
        from dravex.io.tone_gen import fm_tone
        t = fm_tone(1000, 5, 3.0, 0.01, 44100)
        assert len(t) == 441

    def test_harmonic_series(self):
        from dravex.io.tone_gen import harmonic_series
        h = harmonic_series(440, 5, 0.01, 44100)
        assert len(h) == 441
        assert max(abs(s) for s in h) <= 1.001

    def test_click_track(self):
        from dravex.io.tone_gen import click_track
        clicks = click_track(120, 1.0, 44100)
        assert len(clicks) == 44100
