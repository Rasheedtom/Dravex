"""Noise package for dravex signal processing library."""

from dravex.noise.generators import *
from dravex.noise.snr import *
from dravex.noise.denoise import *
from dravex.noise.adaptive import *

def add_white_noise(samples, snr_db=20, rng=None):
    """Convenience: add white noise at target SNR."""
    from dravex.noise.snr import add_noise_at_snr
    return add_noise_at_snr(samples, snr_db, rng)
