"""Tests for noise generation."""
from dravex.noise.generators import white_noise, gaussian_noise, pink_noise
from dravex.noise.snr import estimate_snr, add_noise_at_snr
import math
from dravex.noise.generators import white_noise, gaussian_noise, pink_noise, brown_noise, blue_noise, impulse_noise
from dravex.noise.snr import estimate_snr, add_noise_at_snr, segmental_snr, estimate_snr_single
from dravex.noise.denoise import spectral_subtraction, wiener_filter_denoise
from dravex.noise.adaptive import lms_filter, nlms_filter
from dravex.noise.colored import colored_noise, bandlimited_noise

class TestNoise:
    def test_white_length(self):
        n = white_noise(100)
        assert len(n) == 100

    def test_white_range(self):
        n = white_noise(10000, amplitude=1.0)
        assert all(-1.0 <= s <= 1.0 for s in n)

    def test_gaussian_mean(self):
        n = gaussian_noise(10000, mean=0.0, std=1.0)
        m = sum(n) / len(n)
        assert abs(m) < 0.1

    def test_pink_length(self):
        n = pink_noise(1000)
        assert len(n) == 1000

    def test_snr(self):
        signal = [1.0] * 100
        noise = [0.1] * 100
        snr = estimate_snr(signal, noise)
        assert snr > 10

    def test_add_noise(self):
        signal = [1.0] * 1000
        noisy = add_noise_at_snr(signal, 20)
        assert len(noisy) == 1000



class TestNoiseAdvanced:
    def test_brown_random_walk(self):
        n = brown_noise(1000)
        assert all(-1 <= s <= 1 for s in n)

    def test_blue_noise(self):
        n = blue_noise(1000)
        assert len(n) == 1000

    def test_impulse_noise(self):
        n = impulse_noise(1000, density=0.05)
        nonzero = sum(1 for s in n if abs(s) > 0)
        assert 20 < nonzero < 100

    def test_add_noise_snr(self):
        signal = [math.sin(2*math.pi*100*i/8000) for i in range(8000)]
        noisy = add_noise_at_snr(signal, 20)
        snr = estimate_snr(signal, [n - s for n, s in zip(noisy, signal)])
        assert snr > 15

    def test_segmental_snr(self):
        clean = [1.0] * 1024
        noisy = [1.1] * 1024
        ssnr = segmental_snr(clean, noisy, 256)
        assert ssnr > 10

    def test_snr_single(self):
        x = [0.001] * 1024 + [math.sin(2*math.pi*440*i/8000) for i in range(8192)] + [0.001] * 1024
        snr = estimate_snr_single(x, 1024)
        assert snr > 0

    def test_spectral_subtraction(self):
        signal = [math.sin(2*math.pi*440*i/8000) for i in range(4096)]
        noise = white_noise(4096, 0.1)
        noisy = [s + n for s, n in zip(signal, noise)]
        denoised = spectral_subtraction(noisy, noise[:1024], 512, 128)
        assert len(denoised) > 0

    def test_lms(self):
        desired = [math.sin(0.1*i) + 0.3*math.sin(0.5*i) for i in range(500)]
        reference = [0.3*math.sin(0.5*i) for i in range(500)]
        output, error, w = lms_filter(desired, reference, 16, 0.01)
        assert len(output) == 500

    def test_nlms(self):
        desired = [float(i % 10) for i in range(200)]
        reference = [float((i+1) % 10) for i in range(200)]
        output, w = nlms_filter(desired, reference, 8)
        assert len(output) == 200

    def test_colored_noise(self):
        pink = colored_noise(1024, -1.0)
        assert len(pink) == 1024

    def test_bandlimited(self):
        bl = bandlimited_noise(2048, 500, 2000, 44100)
        assert len(bl) == 2048


class TestColoredNoiseExtra:
    def test_perlin(self):
        from dravex.noise.colored import perlin_noise_1d
        n = perlin_noise_1d(500, 20)
        assert len(n) == 500

    def test_spectral_shaping(self):
        from dravex.noise.colored import spectral_shaping
        from dravex.noise.generators import white_noise
        w = white_noise(256)
        target = [1.0 / (i+1) for i in range(129)]
        shaped = spectral_shaping(w, target)
        assert len(shaped) == 256
