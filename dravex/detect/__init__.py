"""Detect package for dravex signal processing library."""

from dravex.detect.peaks import *
from dravex.detect.onset import *
from dravex.detect.zero_cross import *
from dravex.detect.energy import *
from dravex.detect.silence import *

def find_events(samples, threshold=0.5):
    """Convenience: find above-threshold events."""
    from dravex.detect.peaks import find_peaks
    return find_peaks([abs(s) for s in samples], min_height=threshold)
