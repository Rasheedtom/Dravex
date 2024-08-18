"""Exception hierarchy for dravex signal processing library."""

class DravexError(Exception):
    """Base exception for all dravex errors."""
    def __init__(self, message, code=None, context=None):
        super().__init__(message)
        self.code = code
        self.context = context or {}

class SignalError(DravexError):
    """Error in signal data or operation."""
    pass

class EmptySignalError(SignalError):
    """Operation requires non-empty signal data."""
    def __init__(self):
        super().__init__("Signal is empty", code="EMPTY")

class DimensionError(DravexError):
    """Mismatched signal dimensions or lengths."""
    def __init__(self, expected, got):
        super().__init__(f"Expected length {expected}, got {got}", code="DIM")
        self.expected = expected
        self.got = got

class FilterError(DravexError):
    """Error in filter design or application."""
    pass

class StabilityError(FilterError):
    """Filter is unstable (poles outside unit circle)."""
    def __init__(self, poles=None):
        super().__init__("Unstable filter: poles outside unit circle", code="UNSTABLE")
        self.poles = poles or []

class ConvergenceError(DravexError):
    """Iterative algorithm failed to converge."""
    def __init__(self, message, iterations=0, residual=None):
        super().__init__(message, code="CONVERGENCE")
        self.iterations = iterations
        self.residual = residual

class SampleRateError(DravexError):
    """Invalid or mismatched sample rate."""
    def __init__(self, message="Invalid sample rate"):
        super().__init__(message, code="SAMPLE_RATE")

class QuantizationError(DravexError):
    """Error during signal quantization."""
    def __init__(self, message, bits=None):
        super().__init__(message, code="QUANTIZE")
        self.bits = bits

class WindowError(DravexError):
    """Invalid window parameters."""
    def __init__(self, message):
        super().__init__(message, code="WINDOW")
