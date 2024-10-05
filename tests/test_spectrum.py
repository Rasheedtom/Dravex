"""Tests for spectrum representation."""
from dravex.core.spectrum import Spectrum
import math
from dravex.spectral.power import welch_psd, bartlett_psd, spectral_centroid, dominant_frequency
from dravex.spectral.magnitude import magnitude_spectrum, spectral_peaks, spectral_contrast
from dravex.spectral.phase import phase_spectrum, unwrap_phase, minimum_phase
from dravex.spectral.bandwidth import spectral_bandwidth, spectral_rolloff, spectral_crest, spectral_entropy
from dravex.spectral.formant import find_formants
from dravex.spectral.harmonics import find_harmonics, harmonic_to_noise_ratio
from dravex.spectral.chroma import chroma_vector, detect_chord

class TestSpectrum:
    def test_create(self):
        s = Spectrum([1, 2, 3, 4, 5], sample_rate=44100, n_fft=8)
        assert s.n_bins == 5

    def test_peak_frequency(self):
        mags = [0, 0, 10, 0, 0]
        s = Spectrum(mags, sample_rate=44100, n_fft=8)
        peak = s.peak_frequency()
        assert peak > 0

    def test_power_spectrum(self):
        s = Spectrum([3, 4], sample_rate=44100)
        ps = s.power_spectrum()
        assert abs(ps[0] - 9) < 1e-10
        assert abs(ps[1] - 16) < 1e-10

    def test_spectral_centroid(self):
        s = Spectrum([0, 0, 1, 0, 0], sample_rate=44100, n_fft=8)
        c = s.spectral_centroid()
        assert c > 0

    def test_spectral_flatness(self):
        uniform = Spectrum([1, 1, 1, 1], sample_rate=44100)
        f = uniform.spectral_flatness()
        assert abs(f - 1.0) < 0.01

    def test_band_energy(self):
        s = Spectrum([1]*100, sample_rate=44100, n_fft=198)
        e = s.band_energy(0, 22050)
        assert e > 0

    def test_magnitude_db(self):
        s = Spectrum([1.0, 0.1], sample_rate=44100)
        db = s.magnitude_db()
        assert db[0] > db[1]

    def test_rolloff(self):
        mags = [10, 5, 2, 1, 0]
        s = Spectrum(mags, sample_rate=44100, n_fft=8)
        r = s.spectral_rolloff(0.85)
        assert r >= 0



class TestSpectralAdvanced:
    def test_welch_vs_bartlett(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(8192)]
        f1, p1 = welch_psd(x, 1024, 512, 44100)
        f2, p2 = bartlett_psd(x, 8, 44100)
        assert len(p1) > 0 and len(p2) > 0

    def test_spectral_centroid_sine(self):
        sr = 44100
        x = [math.sin(2*math.pi*1000*i/sr) for i in range(4096)]
        freqs, psd = welch_psd(x, 1024, 512, sr)
        sc = spectral_centroid(psd, freqs)
        assert 500 < sc < 2000

    def test_dominant_frequency(self):
        sr = 44100
        x = [math.sin(2*math.pi*500*i/sr) for i in range(4096)]
        freqs, psd = welch_psd(x, 1024, 512, sr)
        df = dominant_frequency(psd, freqs)
        assert 300 < df < 700

    def test_spectral_peaks(self):
        mags = [0]*10 + [5] + [0]*10 + [8] + [0]*10
        peaks = spectral_peaks(mags, n_peaks=2, min_distance=5)
        assert len(peaks) == 2

    def test_spectral_contrast(self):
        mags = [float(i % 5) for i in range(100)]
        sc = spectral_contrast(mags, 4)
        assert len(sc) == 4

    def test_phase_spectrum(self):
        x = [1, 0, 0, 0, 0, 0, 0, 0]
        p = phase_spectrum(x)
        assert len(p) == 5

    def test_unwrap(self):
        phases = [0.0, 3.0, 6.0, -0.28]
        unwrapped = unwrap_phase(phases)
        for i in range(1, len(unwrapped)):
            diff = abs(unwrapped[i] - unwrapped[i-1])
            assert diff < math.pi + 0.1

    def test_bandwidth(self):
        mags = [0]*10 + [1]*5 + [0]*10
        freqs = list(range(25))
        bw = spectral_bandwidth(mags, freqs)
        assert bw > 0

    def test_rolloff(self):
        mags = list(range(10, 0, -1))
        freqs = list(range(10))
        rf = spectral_rolloff(mags, freqs, 0.85)
        assert rf > 0

    def test_crest(self):
        mags = [0, 0, 10, 0, 0]
        c = spectral_crest(mags)
        assert c > 1

    def test_entropy_uniform(self):
        mags = [1.0] * 10
        e = spectral_entropy(mags)
        assert e > 3

    def test_find_harmonics(self):
        mags = [0] * 100
        mags[10] = 5.0
        mags[20] = 3.0
        mags[30] = 2.0
        harmonics = find_harmonics(mags, 10, 3)
        assert len(harmonics) == 3

    def test_chroma_sine(self):
        mags = [0] * 257
        bin_440 = int(440 * 512 / 44100)
        mags[bin_440] = 10.0
        cv = chroma_vector(mags, 44100, 512)
        assert len(cv) == 12
        assert max(cv) > 0

    def test_detect_chord(self):
        chroma = [0]*12
        chroma[0] = 1.0
        chroma[4] = 0.8
        chroma[7] = 0.6
        chord, score = detect_chord(chroma)
        assert chord == "C"


class TestSpectralMoments:
    def test_spectral_flux(self):
        from dravex.spectral.bandwidth import spectral_flux
        prev = [1.0] * 10
        curr = [2.0] * 10
        flux = spectral_flux(prev, curr)
        assert flux > 0

    def test_spectral_contrast_ratio(self):
        from dravex.spectral.bandwidth import spectral_contrast_ratio
        mags = list(range(60))
        sc = spectral_contrast_ratio(mags, 6)
        assert len(sc) == 6

    def test_spectral_tilt(self):
        from dravex.spectral.bandwidth import spectral_tilt
        mags = [10, 8, 6, 4, 2]
        freqs = [100, 200, 300, 400, 500]
        tilt = spectral_tilt(mags, freqs)
        assert tilt < 0

    def test_spectral_moments(self):
        from dravex.spectral.bandwidth import spectral_moments
        mags = [0, 0, 10, 0, 0]
        freqs = [0, 100, 200, 300, 400]
        moments = spectral_moments(mags, freqs, 3)
        assert len(moments) == 3
        assert abs(moments[0] - 200) < 1e-10


class TestChromaAdvanced:
    def test_pitch_class_histogram(self):
        from dravex.spectral.chroma import pitch_class_histogram
        mags = [0.0] * 257
        mags[10] = 5.0
        hist = pitch_class_histogram(mags, 44100, 512)
        assert len(hist) == 12

    def test_harmonic_change(self):
        from dravex.spectral.chroma import harmonic_change_detection
        chroma1 = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        chroma2 = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        changes = harmonic_change_detection([chroma1, chroma1, chroma2])
        assert changes[2] > changes[1]

    def test_tonnetz(self):
        from dravex.spectral.chroma import tonnetz
        chroma = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        tn = tonnetz(chroma)
        assert len(tn) == 6
