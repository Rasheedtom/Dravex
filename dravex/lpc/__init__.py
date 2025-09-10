"""Lpc package for dravex signal processing library."""

from dravex.lpc.analysis import *
from dravex.lpc.synthesis import *
from dravex.lpc.residual import *
from dravex.lpc.parcor import *

def analyze_lpc(samples, order=12):
    """Convenience: compute LPC coefficients."""
    from dravex.lpc.analysis import lpc_autocorrelation
    return lpc_autocorrelation(samples, order)
