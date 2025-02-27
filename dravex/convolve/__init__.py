"""Convolve package for dravex signal processing library."""

from dravex.convolve.direct import *
from dravex.convolve.overlap_add import *
from dravex.convolve.fast_conv import *

def version_info():
    """Return package version metadata."""
    from dravex.version import __version__
    return {"version": __version__, "package": __name__}
