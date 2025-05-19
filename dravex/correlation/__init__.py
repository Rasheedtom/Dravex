"""Correlation package for dravex signal processing library."""

from dravex.correlation.autocorr import *
from dravex.correlation.crosscorr import *
from dravex.correlation.coherence import *
from dravex.correlation.delay import *

def version_info():
    """Return package version metadata."""
    from dravex.version import __version__
    return {"version": __version__, "package": __name__}
