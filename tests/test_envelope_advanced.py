"""Advanced envelope tests."""
import math
from dravex.core.signal import Signal
from dravex.core.envelope import (amplitude_envelope, rms_envelope, hilbert_envelope,
    attack_decay_sustain_release, smooth_envelope, peak_envelope, temporal_centroid,
    attack_time, decay_time, envelope_follower)

class TestEnvelopeAdvanced:
    def test_amplitude_envelope(self):
        s = Signal([math.sin(2*math.pi*440*i/44100) * (i/44100) for i in range(44100)], 44100)
        env = amplitude_envelope(s, 1024, 512)
        assert len(env) > 0
        assert env[-1] > env[0]

    def test_rms_envelope(self):
        s = Signal([1.0]*4096, 44100)
        env = rms_envelope(s, 1024, 512)
        assert all(abs(e - 1.0) < 0.01 for e in env)

    def test_hilbert_envelope(self):
        s = Signal([math.sin(2*math.pi*440*i/44100) for i in range(2048)], 44100)
        env = hilbert_envelope(s)
        mid = len(env) // 2
        assert abs(env[mid] - 1.0) < 0.2

    def test_adsr(self):
        env = list(range(50)) + list(range(50, 30, -1)) + [30]*50 + list(range(30, 0, -1))
        adsr = attack_decay_sustain_release(env)
        assert adsr["attack"] > 0

    def test_smooth(self):
        env = [0, 10, 0, 10, 0]
        smoothed = smooth_envelope(env, 3)
        assert max(smoothed) < max(env)

    def test_peak_envelope(self):
        x = [math.sin(2*math.pi*100*i/44100) for i in range(4096)]
        env = peak_envelope(x)
        assert len(env) == len(x)

    def test_temporal_centroid(self):
        env = [0]*50 + [1]*50
        tc = temporal_centroid(env)
        assert tc > 50

    def test_attack_time(self):
        env = list(range(100))
        at = attack_time(env, 0.9)
        assert at > 80

    def test_decay_time(self):
        env = list(range(50)) + list(range(50, 0, -1))
        dt = decay_time(env, 0.1)
        assert dt > 0

    def test_envelope_follower(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        env = envelope_follower(x, 10, 100, 44100)
        assert len(env) == len(x)
        assert max(env) > 0
