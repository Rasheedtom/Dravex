"""Core package for dravex signal processing library."""

from dravex.core.signal import *
from dravex.core.frame import *
from dravex.core.spectrum import *
from dravex.core.buffer import *
from dravex.core.channel import *

def load_signal(samples, sample_rate=44100):
    """Convenience: create a Signal from raw samples."""
    from dravex.core.signal import Signal
    return Signal(samples, sample_rate)
