"""Adaptive Differential Pulse-Code Modulation."""

def adpcm_encode(x, bits=4):
    """Simple ADPCM encoder."""
    levels = 2 ** bits
    half = levels // 2
    step_size = 0.1
    predictor = 0.0
    codes = []
    for sample in x:
        diff = sample - predictor
        code = int(round(diff / step_size))
        code = max(-half, min(half - 1, code))
        quantized_diff = code * step_size
        predictor += quantized_diff
        codes.append(code + half)
        error = abs(diff - quantized_diff)
        if error > step_size:
            step_size *= 1.1
        else:
            step_size *= 0.9
        step_size = max(0.001, min(1.0, step_size))
    return codes

def adpcm_decode(codes, bits=4):
    """Simple ADPCM decoder."""
    levels = 2 ** bits
    half = levels // 2
    step_size = 0.1
    predictor = 0.0
    result = []
    for code in codes:
        diff = (code - half) * step_size
        predictor += diff
        result.append(predictor)
        if abs(diff) > step_size * 0.5:
            step_size *= 1.1
        else:
            step_size *= 0.9
        step_size = max(0.001, min(1.0, step_size))
    return result

def dpcm_encode(x):
    """Simple DPCM (non-adaptive)."""
    if not x:
        return []
    diffs = [x[0]]
    for i in range(1, len(x)):
        diffs.append(x[i] - x[i-1])
    return diffs

def dpcm_decode(diffs):
    """DPCM decoder."""
    if not diffs:
        return []
    result = [diffs[0]]
    for i in range(1, len(diffs)):
        result.append(result[-1] + diffs[i])
    return result
