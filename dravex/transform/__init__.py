"""Transform package for dravex signal processing library."""

from dravex.transform.fft import *
from dravex.transform.dct import *
from dravex.transform.hilbert import *
from dravex.transform.stft import *
from dravex.transform.wavelet import *

def compute_spectrum(samples, sample_rate=44100):
    """Convenience: compute magnitude spectrum from samples."""
    from dravex.transform.fft import rfft
    from dravex.core.spectrum import Spectrum
    mags = rfft(samples)
    return Spectrum(mags, sample_rate=sample_rate, n_fft=len(samples))
