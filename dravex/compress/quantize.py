"""Signal quantization utilities."""
import math
from dravex.errors import QuantizationError

def uniform_quantize(x, bits):
    """Uniform mid-tread quantization."""
    if bits < 1:
        raise QuantizationError("Bits must be >= 1", bits=bits)
    levels = 2 ** bits
    half = levels // 2
    result = []
    for s in x:
        q = int(round(s * half))
        q = max(-half, min(half - 1, q))
        result.append(q / half)
    return result

def uniform_dequantize(quantized, bits):
    """Reverse of uniform quantization (identity for already-quantized)."""
    return list(quantized)

def log_quantize(x, bits):
    """Logarithmic quantization (more resolution near zero)."""
    levels = 2 ** bits
    result = []
    for s in x:
        sign = 1.0 if s >= 0 else -1.0
        mag = abs(s)
        log_val = math.log(1 + mag * (levels - 1)) / math.log(levels)
        q = int(round(log_val * (levels // 2)))
        result.append(sign * q / (levels // 2))
    return result

def quantization_noise(x, quantized):
    """Compute quantization noise signal."""
    return [a - b for a, b in zip(x, quantized)]

def sqnr(bits):
    """Signal-to-Quantization-Noise Ratio for uniform quantization."""
    return 6.02 * bits + 1.76

def optimal_bits(target_sqnr_db):
    """Minimum bits for target SQNR."""
    return max(1, math.ceil((target_sqnr_db - 1.76) / 6.02))

def midrise_quantize(x, bits):
    """Mid-rise quantizer (no zero level)."""
    levels = 2 ** bits
    delta = 2.0 / levels
    result = []
    for s in x:
        q = math.floor(s / delta) * delta + delta / 2
        q = max(-1 + delta/2, min(1 - delta/2, q))
        result.append(q)
    return result

def dither_quantize(x, bits, dither_type="triangular"):
    """Quantize with dithering."""
    if dither_type == "rectangular":
        from dravex.compress.dither import rectangular_dither
        return rectangular_dither(x, bits)
    elif dither_type == "triangular":
        from dravex.compress.dither import triangular_dither
        return triangular_dither(x, bits)
    elif dither_type == "noise_shaped":
        from dravex.compress.dither import noise_shaped_dither
        return noise_shaped_dither(x, bits)
    return uniform_quantize(x, bits)

def bit_depth_reduce(x, target_bits, source_bits=None):
    """Reduce bit depth of quantized signal."""
    if source_bits is None:
        source_bits = 16
    shift = source_bits - target_bits
    if shift <= 0:
        return list(x)
    scale_down = 2 ** shift
    return [round(s * (2**(target_bits-1))) / (2**(target_bits-1)) for s in x]

def compute_thd(x, fundamental_freq, sample_rate, n_harmonics=5):
    """Total Harmonic Distortion measurement."""
    import math
    from dravex.transform.goertzel import goertzel_magnitude
    fund_mag = goertzel_magnitude(x, fundamental_freq, sample_rate)
    harm_power = 0.0
    for h in range(2, n_harmonics + 2):
        freq = fundamental_freq * h
        if freq >= sample_rate / 2:
            break
        mag = goertzel_magnitude(x, freq, sample_rate)
        harm_power += mag * mag
    if fund_mag < 1e-12:
        return 0.0
    return math.sqrt(harm_power) / fund_mag * 100
