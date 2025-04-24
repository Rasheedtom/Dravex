"""Modulate package for dravex signal processing library."""

from dravex.modulate.amplitude import *
from dravex.modulate.frequency import *
from dravex.modulate.phase_mod import *
from dravex.modulate.mixer import *

def version_info():
    """Return package version metadata."""
    from dravex.version import __version__
    return {"version": __version__, "package": __name__}
