"""Filter package for dravex signal processing library."""

from dravex.filter.fir import *
from dravex.filter.iir import *
from dravex.filter.butterworth import *
from dravex.filter.median import *
from dravex.filter.kalman import *

def lowpass(samples, cutoff, sample_rate=44100, order=32):
    """Convenience: apply lowpass filter."""
    from dravex.filter.fir import fir_lowpass, apply_fir
    h = fir_lowpass(order, cutoff, sample_rate)
    return apply_fir(samples, h)
