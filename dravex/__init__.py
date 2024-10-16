"""Dravex package for dravex signal processing library."""

from dravex.version import *
from dravex.errors import *
from dravex.config import *
from dravex.constants import *

from dravex.version import __version__, __version_info__

def get_version():
    """Return the current dravex version string."""
    return __version__

def configure(**kwargs):
    """Update global configuration parameters."""
    from dravex.config import Config
    cfg = Config.get()
    for key, value in kwargs.items():
        cfg.set(key, value)
    return cfg
