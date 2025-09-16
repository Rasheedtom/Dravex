"""Integration tests combining multiple modules."""
import math
from dravex.core.signal import Signal
from dravex.transform.fft import fft, ifft, rfft
from dravex.transform.stft import stft, istft, spectrogram
from dravex.filter.fir import fir_lowpass, fir_highpass, apply_fir
from dravex.filter.butterworth import butterworth_lowpass
from dravex.filter.iir import apply_iir
from dravex.window.hann import hann
from dravex.spectral.power import welch_psd
from dravex.spectral.pitch import autocorrelation_pitch
from dravex.noise.generators import white_noise, gaussian_noise
from dravex.noise.snr import add_noise_at_snr, estimate_snr
from dravex.convolve.direct import convolve
from dravex.resample.interpolate import resample_ratio
from dravex.modulate.amplitude import am_modulate
from dravex.io.tone_gen import sine_wave
from dravex.compress.mulaw import mulaw_encode, mulaw_decode
from dravex.detect.peaks import find_peaks

class TestIntegrationPipeline:
    def test_generate_filter_analyze(self):
        """Generate tone, filter, analyze spectrum."""
        x = sine_wave(440, 0.1, 44100)
        h = fir_lowpass(32, 1000, 44100)
        filtered = apply_fir(x, h)
        freqs, psd = welch_psd(filtered, 512, 256, 44100)
        peak_idx = max(range(len(psd)), key=lambda i: psd[i])
        assert 300 < freqs[peak_idx] < 600

    def test_noise_denoise_measure(self):
        """Add noise, denoise, measure SNR."""
        clean = sine_wave(440, 0.2, 44100)
        noisy = add_noise_at_snr(clean, 10)
        h = fir_lowpass(64, 1000, 44100)
        denoised = apply_fir(noisy, h)
        assert len(denoised) == len(clean)

    def test_compress_decompress(self):
        """Compress via mu-law and decompress."""
        x = sine_wave(440, 0.01, 44100)
        encoded = mulaw_encode(x)
        decoded = mulaw_decode(encoded)
        for a, b in zip(x, decoded):
            assert abs(a - b) < 0.05

    def test_resample_and_filter(self):
        """Resample then filter."""
        x = sine_wave(440, 0.1, 44100)
        downsampled = resample_ratio(x, 0.5)
        h = fir_lowpass(32, 500, 22050)
        filtered = apply_fir(downsampled, h)
        assert len(filtered) > 0

    def test_modulate_and_detect(self):
        """AM modulate then detect peaks in envelope."""
        x = sine_wave(100, 0.05, 44100)
        modulated = am_modulate(x, 5000, 44100, 0.8)
        envelope = [abs(s) for s in modulated]
        peaks = find_peaks(envelope, min_distance=20)
        assert len(peaks) > 0

    def test_stft_modify_istft(self):
        """Forward STFT, modify magnitudes, inverse STFT."""
        x = sine_wave(440, 0.1, 44100)
        frames = stft(x, 1024, 256)
        scaled = [(
            [m * 0.5 for m in mags],
            phases
        ) for mags, phases in frames]
        y = istft(scaled, 1024, 256)
        assert len(y) > 0

    def test_convolve_with_impulse_response(self):
        """Convolve signal with room impulse response."""
        x = sine_wave(440, 0.01, 44100)
        ir = [1.0, 0.5, 0.25, 0.125, 0.0625]
        y = convolve(x, ir)
        assert len(y) == len(x) + len(ir) - 1

    def test_butterworth_iir_pipeline(self):
        """Design Butterworth, apply IIR, check passband."""
        b, a = butterworth_lowpass(2, 2000, 44100)
        x = sine_wave(500, 0.1, 44100)
        y = apply_iir(x, b, a)
        in_rms = math.sqrt(sum(s*s for s in x[200:]) / (len(x) - 200))
        out_rms = math.sqrt(sum(s*s for s in y[200:]) / (len(y) - 200))
        assert out_rms > in_rms * 0.5

    def test_pitch_after_noise(self):
        """Detect pitch in noisy signal."""
        clean = sine_wave(220, 0.2, 44100)
        noisy = add_noise_at_snr(clean, 20)
        p = autocorrelation_pitch(noisy, 44100)
        assert 190 < p < 250

    def test_signal_chain(self):
        """Full signal chain: generate -> window -> FFT -> magnitudes."""
        x = sine_wave(1000, 0.05, 44100)
        w = hann(len(x))
        windowed = [s * wi for s, wi in zip(x, w)]
        mags = rfft(windowed)
        peak_bin = max(range(len(mags)), key=lambda i: mags[i])
        freq = peak_bin * 44100 / len(x)
        assert freq > 0


class TestIntegrationMore:
    def test_overlap_buffer(self):
        from dravex.core.buffer import OverlapBuffer
        ob = OverlapBuffer(256, 128)
        for _ in range(10):
            ob.push_frame([0.5] * 256)
        output = ob.get_output()
        assert len(output) > 0

    def test_delay_line(self):
        from dravex.core.buffer import DelayLine
        dl = DelayLine(100)
        for i in range(50):
            dl.write(float(i))
        assert abs(dl.read(1) - 49.0) < 1e-10
        assert abs(dl.read(10) - 40.0) < 1e-10

    def test_delay_line_interpolated(self):
        from dravex.core.buffer import DelayLine
        dl = DelayLine(100)
        dl.write(0.0)
        dl.write(10.0)
        val = dl.read_interpolated(0.5)
        assert 4 < val < 6

    def test_delay_line_feedback(self):
        from dravex.core.buffer import DelayLine
        dl = DelayLine(100)
        results = []
        for i in range(20):
            out = dl.process(1.0 if i == 0 else 0.0, 5, 0.5)
            results.append(out)
        assert results[0] == 0.0
        assert abs(results[5] - 1.0) < 1e-10

    def test_phase_difference(self):
        import math
        from dravex.spectral.phase import phase_difference
        p1 = [0.0, 0.5, 1.0]
        p2 = [0.1, 0.6, 1.1]
        diff = phase_difference(p1, p2)
        assert abs(diff[0] - 0.1) < 1e-10

    def test_phase_vocoder_stretch(self):
        import math
        from dravex.spectral.phase import phase_vocoder_stretch
        prev = [0.0] * 10
        curr = [0.1] * 10
        stretched = phase_vocoder_stretch(prev, curr, 256, 1024, 2.0)
        assert len(stretched) == 10

    def test_transfer_function(self):
        import math
        from dravex.correlation.coherence import transfer_function
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        y = [0.5 * s for s in x]
        freqs, H_mag, H_phase = transfer_function(x, y, 512, 256, 44100)
        assert len(H_mag) > 0

    def test_anderson_darling(self):
        from dravex.stats.distribution import anderson_darling, normal_cdf
        x = list(range(100))
        x = [xi / 100 for xi in x]
        stat = anderson_darling(x, lambda v: v)
        assert stat >= 0

    def test_jarque_bera_normal(self):
        import random
        from dravex.stats.distribution import jarque_bera
        rng = random.Random(42)
        x = [rng.gauss(0, 1) for _ in range(1000)]
        jb = jarque_bera(x)
        assert jb < 10

    def test_chi_squared(self):
        from dravex.stats.distribution import chi_squared_test
        observed = [10, 20, 30]
        expected = [15, 15, 30]
        chi2 = chi_squared_test(observed, expected)
        assert chi2 > 0

    def test_empirical_cdf(self):
        from dravex.stats.distribution import empirical_cdf
        x = [1, 2, 3, 4, 5]
        assert abs(empirical_cdf(x, 3) - 0.6) < 1e-10
        assert abs(empirical_cdf(x, 0) - 0.0) < 1e-10

    def test_exact_blackman(self):
        from dravex.window.blackman import exact_blackman
        w = exact_blackman(64)
        assert len(w) == 64

    def test_window_mainlobe(self):
        from dravex.window.blackman import window_mainlobe_width
        from dravex.window.hann import hann
        width = window_mainlobe_width(hann, 256, 44100)
        assert width > 0

    def test_mel_spectrogram_shape(self):
        import math
        from dravex.transform.stft import mel_spectrogram
        x = [math.sin(2*math.pi*440*i/44100) for i in range(8192)]
        ms = mel_spectrogram(x, 44100, 1024, 256, 64)
        assert len(ms) > 0
        assert len(ms[0]) == 64

    def test_pitch_strength(self):
        import math
        from dravex.spectral.pitch import pitch_strength_metric
        x = [math.sin(2*math.pi*200*i/44100) for i in range(4096)]
        strength = pitch_strength_metric(x, 44100)
        assert strength > 0.5

    def test_multi_pitch(self):
        import math
        from dravex.spectral.pitch import multi_pitch
        x = [math.sin(2*math.pi*440*i/44100) + math.sin(2*math.pi*880*i/44100) for i in range(8192)]
        pitches = multi_pitch(x, 44100, 3)
        assert len(pitches) >= 1

    def test_snr_per_band(self):
        import math
        from dravex.noise.snr import snr_per_band
        clean = [math.sin(2*math.pi*440*i/44100) for i in range(8192)]
        from dravex.noise.generators import white_noise
        noise = white_noise(8192, 0.1)
        noisy = [c + n for c, n in zip(clean, noise)]
        bands = snr_per_band(clean, noisy, 4, 44100)
        assert len(bands) == 4

    def test_lucy_richardson(self):
        from dravex.convolve.deconvolve import lucy_richardson
        h = [0.5, 1.0, 0.5]
        x = [0, 0, 1, 0, 0, 0, 0, 0]
        from dravex.convolve.direct import convolve
        y = convolve(x, h)[:8]
        # Just verify it runs without error
        recovered = lucy_richardson(y, h, 5)
        assert len(recovered) == 8

    def test_end_to_end_voice(self):
        """Full voice processing pipeline."""
        import math
        from dravex.io.tone_gen import sine_wave
        from dravex.core.normalize import pre_emphasis
        from dravex.core.signal import Signal
        from dravex.lpc.analysis import lpc_autocorrelation, lpc_error
        from dravex.lpc.synthesis import lpc_synthesize
        voice = sine_wave(200, 0.05, 8000)
        sig = Signal(voice, 8000)
        emphasized = pre_emphasis(sig, 0.97)
        coeffs = lpc_autocorrelation(emphasized.samples, 10)
        residual = lpc_error(emphasized.samples, coeffs)
        reconstructed = lpc_synthesize(residual, coeffs)
        assert len(reconstructed) == len(voice)
