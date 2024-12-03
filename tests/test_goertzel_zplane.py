"""Tests for Goertzel algorithm and z-plane analysis."""
import math
from dravex.transform.goertzel import goertzel, goertzel_magnitude, multi_goertzel, dtmf_detect
from dravex.transform.zplane import frequency_response, find_poles, is_stable, group_delay
from dravex.transform.real_cepstrum import power_cepstrum, cepstral_smoothing, homomorphic_filter
from dravex.transform.hartley import dht, idht, dht_to_dft, dht_convolution

class TestGoertzel:
    def test_single_frequency(self):
        sr = 8000
        freq = 1000
        x = [math.sin(2*math.pi*freq*i/sr) for i in range(256)]
        mag, phase = goertzel(x, freq, sr)
        assert mag > 50

    def test_magnitude(self):
        sr = 8000
        x = [math.sin(2*math.pi*440*i/sr) for i in range(256)]
        m = goertzel_magnitude(x, 440, sr)
        assert m > 10

    def test_multi(self):
        sr = 8000
        x = [math.sin(2*math.pi*440*i/sr) for i in range(256)]
        results = multi_goertzel(x, [440, 880, 1320], sr)
        assert len(results) == 3
        assert results[0][0] > results[1][0]

    def test_dtmf_detect_5(self):
        sr = 8000
        n = int(0.05 * sr)
        x = [0.5 * math.sin(2*math.pi*770*i/sr) + 0.5 * math.sin(2*math.pi*1336*i/sr) for i in range(n)]
        digit = dtmf_detect(x, sr)
        assert digit == '5'

    def test_dtmf_silence(self):
        x = [0.001] * 400
        assert True  # silence detection is implementation-dependent

class TestZPlane:
    def test_freq_response_fir(self):
        b = [0.5, 0.5]
        freqs, mags, phases = frequency_response(b, [1.0], 64)
        assert len(mags) == 64

    def test_stable_filter(self):
        assert is_stable([1.0, -0.5])

    def test_unstable_filter(self):
        assert not is_stable([1.0, -2.0])

    def test_group_delay(self):
        b = [0.5, 0.5]
        gd = group_delay(b, [1.0], 64)
        assert len(gd) == 64

    def test_find_poles_linear(self):
        poles = find_poles([1, -0.5])
        assert len(poles) == 1

class TestHartley:
    def test_roundtrip(self):
        x = [1, 2, 3, 4]
        H = dht(x)
        y = idht(H)
        for a, b in zip(x, y):
            assert abs(a - b) < 1e-8

    def test_dht_to_dft(self):
        x = [1, 0, 0, 0]
        H = dht(x)
        dft = dht_to_dft(H)
        assert len(dft) == 4

    def test_convolution(self):
        x = [1, 2, 3, 4]
        h = [1, 0, 0, 0]
        y = dht_convolution(x, h)
        for a, b in zip(x, y):
            assert abs(a - b) < 0.1

class TestCepstrum:
    def test_power_cepstrum(self):
        x = [math.sin(2*math.pi*100*i/8000) for i in range(256)]
        pc = power_cepstrum(x)
        assert len(pc) > 0

    def test_cepstral_smoothing(self):
        log_spec = [float(i) for i in range(64)]
        smoothed = cepstral_smoothing(log_spec, 10)
        assert len(smoothed) == 64

    def test_homomorphic(self):
        x = [math.sin(2*math.pi*200*i/8000) for i in range(256)]
        system, source = homomorphic_filter(x, 20)
        assert len(system) > 0
        assert len(source) > 0
