"""LPC synthesis (all-pole filter).

Reconstructs signal by driving an all-pole filter
with the LPC residual (excitation) signal.
"""

def lpc_synthesize(residual, coeffs):
    """Synthesize signal from LPC coefficients and residual."""
    n = len(residual)
    order = len(coeffs) - 1
    output = [0.0] * n
    for i in range(n):
        prediction = 0.0
        for k in range(1, min(order + 1, i + 1)):
            prediction -= coeffs[k] * output[i - k]
        output[i] = residual[i] + prediction
    return output

def lpc_resynthesize(x, order=12):
    """Analyze then resynthesize (reconstruction check)."""
    from dravex.lpc.analysis import lpc_autocorrelation, lpc_error
    coeffs = lpc_autocorrelation(x, order)
    residual = lpc_error(x, coeffs)
    return lpc_synthesize(residual, coeffs)

def pitch_modify(residual, coeffs, pitch_factor):
    """Modify pitch by resampling residual."""
    from dravex.resample.interpolate import resample_ratio
    resampled = resample_ratio(residual, 1.0 / pitch_factor)
    target_len = len(residual)
    if len(resampled) > target_len:
        resampled = resampled[:target_len]
    else:
        resampled = resampled + [0.0] * (target_len - len(resampled))
    return lpc_synthesize(resampled, coeffs)

def formant_shift(x, order=12, shift_factor=1.2):
    """Shift formants by modifying LPC coefficients."""
    from dravex.lpc.analysis import lpc_autocorrelation, lpc_error
    import math
    coeffs = lpc_autocorrelation(x, order)
    residual = lpc_error(x, coeffs)
    shifted = list(coeffs)
    for i in range(1, len(shifted)):
        shifted[i] *= shift_factor ** (-i)
    return lpc_synthesize(residual, shifted)
