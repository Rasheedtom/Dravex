"""Tests for configuration and error handling."""
import pytest
from dravex.config import Config
from dravex.errors import (DravexError, SignalError, EmptySignalError, DimensionError,
    FilterError, StabilityError, ConvergenceError, SampleRateError, QuantizationError, WindowError)
from dravex.core.signal import Signal
from dravex.core.buffer import RingBuffer
from dravex.version import __version__, __version_info__
import dravex

class TestConfig:
    def test_singleton(self):
        Config.reset()
        c1 = Config.get()
        c2 = Config.get()
        assert c1 is c2
        Config.reset()

    def test_set_get(self):
        Config.reset()
        cfg = Config.get()
        cfg.set("epsilon", 0.001)
        assert cfg.epsilon == 0.001
        Config.reset()

    def test_unknown_key_raises(self):
        Config.reset()
        cfg = Config.get()
        with pytest.raises(KeyError):
            cfg.set("nonexistent_key", 42)
        Config.reset()

    def test_unknown_attr_raises(self):
        Config.reset()
        cfg = Config.get()
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_key
        Config.reset()

    def test_snapshot_restore(self):
        Config.reset()
        cfg = Config.get()
        snap = cfg.snapshot()
        cfg.set("fft_size", 2048)
        cfg.restore(snap)
        assert cfg.fft_size == 1024
        Config.reset()

    def test_default_values(self):
        Config.reset()
        cfg = Config.get()
        assert cfg.default_sample_rate == 44100
        assert cfg.fft_size == 1024
        assert cfg.hop_size == 256
        Config.reset()

class TestErrors:
    def test_dravex_error(self):
        e = DravexError("test", code="TEST", context={"key": "val"})
        assert str(e) == "test"
        assert e.code == "TEST"

    def test_empty_signal_error(self):
        e = EmptySignalError()
        assert e.code == "EMPTY"

    def test_dimension_error(self):
        e = DimensionError(10, 5)
        assert e.expected == 10
        assert e.got == 5

    def test_stability_error(self):
        e = StabilityError(poles=[(1.5, 0)])
        assert len(e.poles) == 1

    def test_convergence_error(self):
        e = ConvergenceError("did not converge", iterations=100, residual=0.01)
        assert e.iterations == 100

    def test_sample_rate_error(self):
        e = SampleRateError()
        assert e.code == "SAMPLE_RATE"

    def test_quantization_error(self):
        e = QuantizationError("bad bits", bits=0)
        assert e.bits == 0

    def test_window_error(self):
        e = WindowError("bad params")
        assert e.code == "WINDOW"

class TestBufferErrors:
    def test_empty_pop(self):
        b = RingBuffer(5)
        with pytest.raises(EmptySignalError):
            b.pop()

    def test_peek_out_of_range(self):
        b = RingBuffer(5)
        b.push(1.0)
        with pytest.raises(IndexError):
            b.peek(5)



class TestVersion:
    def test_version_string(self):
        assert isinstance(__version__, str)
        assert '.' in __version__

    def test_version_info(self):
        assert len(__version_info__) == 3
        assert all(isinstance(v, int) for v in __version_info__)

    def test_package_version(self):
        assert dravex.__version__ == __version__

    def test_get_version(self):
        assert dravex.get_version() == __version__

    def test_configure(self):
        from dravex.config import Config
        Config.reset()
        dravex.configure(epsilon=0.001)
        cfg = Config.get()
        assert cfg.epsilon == 0.001
        Config.reset()


class TestNormalizeDynamics:
    def test_auto_gain(self):
        from dravex.core.normalize import auto_gain_control
        s = Signal([0.01] * 1000, 44100)
        agc = auto_gain_control(s, target_rms=0.1)
        assert agc.rms() > s.rms()

    def test_compressor(self):
        import math
        from dravex.core.normalize import compressor
        loud = Signal([math.sin(2*math.pi*440*i/44100) for i in range(4096)], 44100)
        compressed = compressor(loud, threshold_db=-10, ratio=4.0)
        assert compressed.max_amplitude() <= loud.max_amplitude() + 0.01

    def test_limiter(self):
        from dravex.core.normalize import limiter
        s = Signal([2.0, -2.0, 0.5], 44100)
        limited = limiter(s, 0.95)
        assert limited.max_amplitude() <= 0.95 + 0.01

    def test_noise_gate(self):
        from dravex.core.normalize import noise_gate
        quiet = [0.001] * 1000
        loud = [0.5] * 1000
        s = Signal(quiet + loud + quiet, 44100)
        gated = noise_gate(s, threshold_db=-20)
        assert sum(abs(x) for x in gated.samples[:500]) < sum(abs(x) for x in gated.samples[1000:1500])
