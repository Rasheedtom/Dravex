"""Phase modulation."""
import math
from dravex.constants import TAU

def pm_modulate(signal, carrier_freq, phase_deviation, sample_rate):
    """Phase modulation: y = cos(2*pi*fc*t + beta*x(t))."""
    result = []
    for i in range(len(signal)):
        t = i / sample_rate
        result.append(math.cos(TAU * carrier_freq * t + phase_deviation * signal[i]))
    return result

def pm_demodulate(modulated, carrier_freq, sample_rate):
    """Phase demodulation via Hilbert transform."""
    from dravex.transform.hilbert import instantaneous_phase
    phase = instantaneous_phase(modulated)
    expected = [TAU * carrier_freq * i / sample_rate for i in range(len(modulated))]
    return [p - e for p, e in zip(phase, expected)]

def bpsk_modulate(bits, carrier_freq, samples_per_bit, sample_rate):
    """Binary Phase-Shift Keying modulation."""
    result = []
    for bit in bits:
        phase = 0 if bit else math.pi
        for i in range(samples_per_bit):
            t = i / sample_rate
            result.append(math.cos(TAU * carrier_freq * t + phase))
    return result

def bpsk_demodulate(signal, carrier_freq, samples_per_bit, sample_rate):
    """BPSK demodulation via correlation."""
    n_bits = len(signal) // samples_per_bit
    bits = []
    for b in range(n_bits):
        corr = 0.0
        for i in range(samples_per_bit):
            t = i / sample_rate
            idx = b * samples_per_bit + i
            corr += signal[idx] * math.cos(TAU * carrier_freq * t)
        bits.append(1 if corr > 0 else 0)
    return bits

def dpsk_encode(bits):
    """Differential encoding for DPSK."""
    encoded = [bits[0]]
    for i in range(1, len(bits)):
        encoded.append(encoded[-1] ^ bits[i])
    return encoded
