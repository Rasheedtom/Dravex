"""Edge case tests for coverage."""
from dravex.core.signal import Signal
from dravex.core.buffer import RingBuffer
from dravex.errors import EmptySignalError, DimensionError, QuantizationError
from dravex.config import Config
from dravex.constants import next_power_of_2, sinc, amplitude_to_db, hz_to_mel
import pytest

class TestEdgeCoverage:
    def test_empty_signal_error(self):
        s = Signal([], 44100)
        with pytest.raises(EmptySignalError):
            s.max_amplitude()

    def test_dimension_error(self):
        a = Signal([1, 2], 44100)
        b = Signal([1], 44100)
        with pytest.raises(DimensionError):
            a + b

    def test_config_reset(self):
        Config.reset()
        cfg = Config.get()
        assert cfg.epsilon == 1e-12
        Config.reset()

    def test_config_unknown(self):
        cfg = Config.get()
        with pytest.raises(KeyError):
            cfg.set("nonexistent", 1)
        Config.reset()

    def test_config_snapshot_restore(self):
        cfg = Config.get()
        snap = cfg.snapshot()
        cfg.set("epsilon", 0.001)
        cfg.restore(snap)
        assert cfg.epsilon == 1e-12
        Config.reset()

    def test_buffer_empty_pop(self):
        b = RingBuffer(5)
        with pytest.raises(EmptySignalError):
            b.pop()

    def test_next_power_of_2(self):
        assert next_power_of_2(1) == 1
        assert next_power_of_2(3) == 4
        assert next_power_of_2(8) == 8

    def test_sinc_zero(self):
        assert abs(sinc(0) - 1.0) < 1e-10

    def test_amplitude_to_db_zero(self):
        assert amplitude_to_db(0) == -120.0

    def test_hz_to_mel(self):
        assert hz_to_mel(0) == 0.0
        assert hz_to_mel(1000) > 0

    def test_quantization_error(self):
        with pytest.raises(QuantizationError):
            from dravex.compress.quantize import uniform_quantize
            uniform_quantize([0.5], 0)

    def test_signal_zeros(self):
        s = Signal.zeros(10)
        assert s.rms() == 0.0

    def test_signal_negate_add(self):
        s = Signal([1, 2, 3], 44100)
        n = -s
        z = s + n
        assert all(abs(x) < 1e-10 for x in z.samples)


class TestEdgeCoverageAdditional:
    def test_signal_split(self):
        s = Signal([float(i) for i in range(100)], 44100)
        parts = s.split(5)
        assert len(parts) == 5

    def test_complex_signal_normalize(self):
        from dravex.core.complex_signal import ComplexSignal
        cs = ComplexSignal([3, 0], [4, 0])
        n = cs.normalize()
        assert abs(n.energy() - 1.0) < 0.01

    def test_complex_signal_rotate(self):
        import math
        from dravex.core.complex_signal import ComplexSignal
        cs = ComplexSignal([1], [0])
        rotated = cs.rotate(math.pi / 2)
        assert abs(rotated.real[0]) < 0.01
        assert abs(rotated.imag[0] - 1.0) < 0.01

    def test_mfcc_full(self):
        import math
        from dravex.spectral.mfcc import mfcc_full
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        frames = mfcc_full(x, 44100, 13, 26, 512, 256)
        assert len(frames) > 0
        assert len(frames[0]) == 13

    def test_mfcc_features(self):
        import math
        from dravex.spectral.mfcc import mfcc_features
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        feats = mfcc_features(x, 44100, 13)
        assert len(feats) > 0
        assert len(feats[0]) == 39

    def test_tristimulus(self):
        from dravex.spectral.harmonics import tristimulus
        harmonics = [(10, 5.0), (20, 3.0), (30, 2.0), (40, 1.0), (50, 0.5)]
        t1, t2, t3 = tristimulus(harmonics)
        assert abs(t1 + t2 + t3 - 1.0) < 0.01

    def test_pitch_contour(self):
        import math
        from dravex.spectral.pitch import pitch_contour
        x = [math.sin(2*math.pi*200*i/44100) for i in range(16384)]
        contour = pitch_contour(x, 44100, 2048, 512)
        assert len(contour) > 0
        assert len(contour) > 0

    def test_resonator(self):
        import math
        from dravex.filter.iir import resonator
        x = [0.0] * 1024
        x[0] = 1.0
        y = resonator(x, 1000, 100, 44100)
        assert len(y) == 1024
        assert max(abs(s) for s in y) > 0

    def test_parametric_eq(self):
        import math
        from dravex.filter.iir import parametric_eq
        x = [math.sin(2*math.pi*1000*i/44100) for i in range(2048)]
        boosted = parametric_eq(x, 1000, 6, 2, 44100)
        assert len(boosted) == 2048

    def test_sos_filter(self):
        from dravex.filter.iir import sos_filter
        x = [1.0] + [0.0] * 49
        sections = [(1, 0, 0, 1, 0, 0)]
        y = sos_filter(x, sections)
        assert abs(y[0] - 1.0) < 1e-10

    def test_sliding_goertzel(self):
        import math
        from dravex.transform.goertzel import sliding_goertzel
        x = [math.sin(2*math.pi*440*i/8000) for i in range(2048)]
        results = sliding_goertzel(x, 440, 8000, 256, 128)
        assert len(results) > 0
        assert all(r > 0 for r in results)

    def test_tone_detect(self):
        import math
        from dravex.transform.goertzel import tone_detect
        sr = 8000
        x = [math.sin(2*math.pi*440*i/sr) + math.sin(2*math.pi*880*i/sr) for i in range(512)]
        detected = tone_detect(x, [440, 660, 880, 1000], sr, 0.3)
        assert 440 in detected
        assert 880 in detected

    def test_peak_snr(self):
        from dravex.noise.snr import peak_snr
        clean = [1.0] * 100
        noisy = [1.01] * 100
        psnr = peak_snr(clean, noisy)
        assert psnr > 30

    def test_si_sdr(self):
        import math
        from dravex.noise.snr import si_sdr
        clean = [math.sin(0.1*i) for i in range(200)]
        noisy = [s + 0.01 for s in clean]
        sdr = si_sdr(clean, noisy)
        assert sdr > 10

    def test_regularized_deconv(self):
        from dravex.convolve.deconvolve import regularized_deconvolution
        h = [1, 0.5, 0.25, 0, 0, 0, 0, 0]
        x = [1, 0, 0, 0, 0, 0, 0, 0]
        from dravex.convolve.overlap_add import fft_convolve
        y = fft_convolve(x, h)[:8]
        recovered = regularized_deconvolution(y, h, 0.001)
        assert abs(recovered[0] - 1.0) < 0.5

    def test_dct_iv(self):
        from dravex.transform.dct import dct_iv
        x = [1, 2, 3, 4]
        result = dct_iv(x)
        assert len(result) == 4

    def test_mdct_imdct(self):
        from dravex.transform.dct import mdct, imdct
        x = [float(i) for i in range(8)]
        X = mdct(x)
        assert len(X) == 4

    def test_frame_features(self):
        from dravex.core.frame import frame_features
        feats = frame_features([1, -1, 1, -1, 1])
        assert "energy" in feats
        assert "zcr" in feats
        assert feats["zcr"] == 1.0

    def test_reconstruct_frames(self):
        from dravex.core.frame import overlap_frames, reconstruct_from_frames
        x = [float(i)/100 for i in range(1000)]
        from dravex.window.hann import hann
        w = hann(256)
        frames, positions = overlap_frames(x, 256, 128, w)
        y = reconstruct_from_frames(frames, positions, 256, w)
        for i in range(200, 800):
            assert abs(x[i] - y[i]) < 0.2

    def test_formant_track(self):
        import math
        from dravex.spectral.formant import formant_track
        x = [math.sin(2*math.pi*500*i/8000) for i in range(4096)]
        tracks = formant_track(x, 8000, 512, 256, 2)
        assert len(tracks) == 2

    def test_pan_stereo(self):
        from dravex.modulate.mixer import pan_stereo
        mono = [1.0] * 10
        left, right = pan_stereo(mono, -1.0)
        assert sum(abs(l) for l in left) > sum(abs(r) for r in right)

    def test_stereo_width(self):
        from dravex.modulate.mixer import stereo_width
        left = [1.0] * 10
        right = [0.0] * 10
        nl, nr = stereo_width(left, right, 2.0)
        assert len(nl) == 10

    def test_cascade_decimate(self):
        from dravex.resample.decimate import cascade_decimate
        x = [float(i)/100 for i in range(4096)]
        y = cascade_decimate(x, 4, 44100)
        assert abs(len(y) - 1024) < 100

    def test_residual_spectrum(self):
        import math
        from dravex.lpc.residual import residual_spectrum
        r = [math.sin(0.5*i) for i in range(256)]
        spec = residual_spectrum(r, 256)
        assert len(spec) > 0

    def test_pitch_from_residual(self):
        import math
        from dravex.lpc.residual import pitch_from_residual
        period = 40
        residual = [0.0] * 320
        for i in range(0, 320, period):
            residual[i] = 1.0
        p = pitch_from_residual(residual, 8000, 50, 400)
        assert 150 < p < 250

    def test_spectral_tilt(self):
        import math
        from dravex.lpc.residual import spectral_tilt
        r = [math.sin(0.1*i) for i in range(256)]
        tilt = spectral_tilt(r, 256)
        assert isinstance(tilt, float)


class TestFadeEnvelope:
    def test_fade_envelope(self):
        from dravex.core.normalize import fade_envelope
        s = Signal([1.0] * 44100, 44100)
        faded = fade_envelope(s, 10, 50, 0.7, 100)
        assert faded[0] < 0.1
        assert faded[-1] < 0.1
        mid = faded[22050]
        assert abs(mid - 0.7) < 0.05


class TestCoverageGaps:
    def test_signal_len(self):
        s = Signal([1, 2, 3], 44100)
        assert len(s) == 3

    def test_config_restore_unknown_key(self):
        from dravex.config import Config
        Config.reset()
        cfg = Config.get()
        cfg.restore({'nonexistent': 42, 'epsilon': 0.1})
        assert cfg.epsilon == 0.1
        Config.reset()

    def test_white_noise_burst(self):
        from dravex.io.tone_gen import white_noise_burst
        burst = white_noise_burst(0.01, 1.0, 44100)
        assert len(burst) == 441
