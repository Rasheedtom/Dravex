"""Tests for constants and utility functions."""
import math
from dravex.constants import (
    PI, TAU, SQRT2, LN2, amplitude_to_db, db_to_amplitude, power_to_db, db_to_power,
    hz_to_mel, mel_to_hz, hz_to_midi, midi_to_hz, hz_to_bark,
    next_power_of_2, is_power_of_2, sinc, rms, DEFAULT_EPSILON
)

class TestConstants:
    def test_pi(self):
        assert abs(PI - math.pi) < 1e-15

    def test_tau(self):
        assert abs(TAU - 2 * math.pi) < 1e-15

    def test_sqrt2(self):
        assert abs(SQRT2 - math.sqrt(2)) < 1e-15

    def test_ln2(self):
        assert abs(LN2 - math.log(2)) < 1e-15

class TestDBConversion:
    def test_amplitude_to_db(self):
        assert abs(amplitude_to_db(1.0) - 0.0) < 1e-10

    def test_amplitude_to_db_half(self):
        db = amplitude_to_db(0.5)
        assert abs(db - (-6.02)) < 0.1

    def test_db_to_amplitude(self):
        assert abs(db_to_amplitude(0.0) - 1.0) < 1e-10

    def test_amplitude_roundtrip(self):
        for amp in [0.1, 0.5, 1.0, 2.0]:
            db = amplitude_to_db(amp)
            assert abs(db_to_amplitude(db) - amp) < 1e-8

    def test_power_to_db(self):
        assert abs(power_to_db(1.0) - 0.0) < 1e-10

    def test_power_roundtrip(self):
        for p in [0.01, 0.1, 1.0, 10.0]:
            db = power_to_db(p)
            assert abs(db_to_power(db) - p) < 1e-8

    def test_zero_amplitude(self):
        assert amplitude_to_db(0.0) == -120.0

class TestFreqConversion:
    def test_mel_roundtrip(self):
        for hz in [100, 500, 1000, 5000]:
            mel = hz_to_mel(hz)
            assert abs(mel_to_hz(mel) - hz) < 0.1

    def test_midi_roundtrip(self):
        for hz in [220, 440, 880]:
            midi = hz_to_midi(hz)
            assert abs(midi_to_hz(midi) - hz) < 0.1

    def test_a440_midi(self):
        assert abs(hz_to_midi(440) - 69) < 1e-10

    def test_bark(self):
        b = hz_to_bark(1000)
        assert 5 < b < 15

class TestUtilities:
    def test_next_pow2(self):
        assert next_power_of_2(1) == 1
        assert next_power_of_2(3) == 4
        assert next_power_of_2(7) == 8
        assert next_power_of_2(8) == 8
        assert next_power_of_2(9) == 16

    def test_is_pow2(self):
        assert is_power_of_2(1)
        assert is_power_of_2(2)
        assert is_power_of_2(256)
        assert not is_power_of_2(3)
        assert not is_power_of_2(0)

    def test_sinc_zero(self):
        assert abs(sinc(0) - 1.0) < 1e-10

    def test_sinc_integer(self):
        for n in [1, 2, 3, 5]:
            assert abs(sinc(n)) < 1e-10

    def test_rms_simple(self):
        assert abs(rms([3, 4]) - math.sqrt(12.5)) < 1e-10

    def test_rms_sine(self):
        x = [math.sin(2*math.pi*i/100) for i in range(100)]
        r = rms(x)
        assert abs(r - 1/math.sqrt(2)) < 0.02

    def test_rms_empty(self):
        assert rms([]) == 0.0
