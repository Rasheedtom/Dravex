"""Tests for modulation."""
import math
from dravex.modulate.amplitude import am_modulate, ring_modulate
from dravex.modulate.frequency import fm_modulate, chirp
from dravex.modulate.amplitude import am_modulate, am_demodulate, dsb_sc_modulate, tremolo, ssb_modulate
from dravex.modulate.frequency import fm_modulate, fm_demodulate, vibrato, chirp
from dravex.modulate.phase_mod import pm_modulate, bpsk_modulate, bpsk_demodulate
from dravex.modulate.mixer import mix, frequency_shift, crossfade
from dravex.modulate.qam import qam16_modulate, qam16_demodulate, symbol_error_rate

class TestModulation:
    def test_am(self):
        x = [1.0] * 100
        y = am_modulate(x, 1000, 44100, 1.0)
        assert len(y) == 100

    def test_ring_mod(self):
        a = [1, 2, 3]
        b = [2, 3, 4]
        r = ring_modulate(a, b)
        assert r == [2, 6, 12]

    def test_fm(self):
        x = [0.0] * 100
        y = fm_modulate(x, 1000, 5.0, 44100)
        assert len(y) == 100

    def test_chirp(self):
        c = chirp(0.01, 100, 1000, 44100)
        assert len(c) == 441

    def test_am_envelope(self):
        x = [math.sin(2*math.pi*10*i/1000) for i in range(1000)]
        y = am_modulate(x, 100, 1000, 0.5)
        assert max(abs(s) for s in y) > 0



class TestModulateAdvanced:
    def test_am_roundtrip(self):
        x = [math.sin(2*math.pi*100*i/44100) for i in range(4096)]
        modulated = am_modulate(x, 5000, 44100, 0.8)
        demod = am_demodulate(modulated, 5000, 44100)
        assert len(demod) == len(x)

    def test_dsb_sc(self):
        x = [1.0] * 100
        y = dsb_sc_modulate(x, 1000, 44100)
        assert len(y) == 100

    def test_tremolo(self):
        x = [1.0] * 44100
        y = tremolo(x, 5, 0.5, 44100)
        assert min(y) < 1.0

    def test_ssb(self):
        x = [math.sin(2*math.pi*500*i/44100) for i in range(2048)]
        upper = ssb_modulate(x, 5000, 44100, True)
        assert len(upper) == len(x)

    def test_vibrato(self):
        x = [math.sin(2*math.pi*440*i/44100) for i in range(4096)]
        y = vibrato(x, 5, 10, 44100)
        assert len(y) == len(x)

    def test_chirp_length(self):
        c = chirp(0.5, 100, 5000, 44100)
        assert abs(len(c) - 22050) < 2

    def test_bpsk_roundtrip(self):
        bits = [1, 0, 1, 1, 0]
        mod = bpsk_modulate(bits, 1000, 100, 44100)
        demod = bpsk_demodulate(mod, 1000, 100, 44100)
        assert demod == bits

    def test_mix_two(self):
        a = [1.0] * 10
        b = [2.0] * 10
        m = mix([a, b])
        assert all(abs(s - 1.5) < 1e-10 for s in m)

    def test_crossfade(self):
        a = [1.0] * 10
        b = [0.0] * 10
        cf = crossfade(a, b, 10)
        assert abs(cf[0] - 1.0) < 1e-10
        assert abs(cf[9] - 0.1) < 0.15

    def test_qam_roundtrip(self):
        symbols = [0, 5, 10, 15]
        mod = qam16_modulate(symbols, 1000, 200, 44100)
        demod = qam16_demodulate(mod, 1000, 200, 44100)
        assert demod == symbols

    def test_symbol_error_rate(self):
        tx = [1, 2, 3, 4]
        rx = [1, 2, 0, 4]
        ser = symbol_error_rate(tx, rx)
        assert abs(ser - 0.25) < 1e-10
