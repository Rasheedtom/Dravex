"""Compress package for dravex signal processing library."""

from dravex.compress.mulaw import *
from dravex.compress.alaw import *
from dravex.compress.adpcm import *
from dravex.compress.quantize import *

def version_info():
    """Return package version metadata."""
    from dravex.version import __version__
    return {"version": __version__, "package": __name__}
