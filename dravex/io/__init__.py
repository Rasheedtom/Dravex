"""Io package for dravex signal processing library."""

from dravex.io.wav import *
from dravex.io.raw import *
from dravex.io.csv_io import *
from dravex.io.tone_gen import *

def version_info():
    """Return package version metadata."""
    from dravex.version import __version__
    return {"version": __version__, "package": __name__}
