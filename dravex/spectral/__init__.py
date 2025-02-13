"""Spectral package for dravex signal processing library."""

from dravex.spectral.power import *
from dravex.spectral.magnitude import *
from dravex.spectral.phase import *
from dravex.spectral.pitch import *
from dravex.spectral.mfcc import *

def analyze(samples, sample_rate=44100, frame_size=1024):
    """Quick spectral analysis returning key features."""
    from dravex.transform.fft import rfft
    mags = rfft(samples[:frame_size])
    freqs = [i * sample_rate / frame_size for i in range(len(mags))]
    from dravex.spectral.bandwidth import spectral_centroid as sc
    return {"centroid": sc(mags, freqs), "peak_mag": max(mags)}
