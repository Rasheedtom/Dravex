"""Resample package for dravex signal processing library."""

from dravex.resample.interpolate import *
from dravex.resample.decimate import *
from dravex.resample.polyphase import *
from dravex.resample.sinc_resample import *

def version_info():
    """Return package version metadata."""
    from dravex.version import __version__
    return {"version": __version__, "package": __name__}
