"""Window package for dravex signal processing library."""

from dravex.window.hamming import *
from dravex.window.hann import *
from dravex.window.blackman import *
from dravex.window.kaiser import *
from dravex.window.gaussian import *

def get_window(name, n):
    """Get window function by name."""
    windows = {"hamming": hamming, "hann": hann, "blackman": blackman}
    func = windows.get(name)
    if func is None:
        raise ValueError(f"Unknown window: {name}")
    return func(n)
