"""Quadrature Amplitude Modulation."""
import math
from dravex.constants import TAU

def qam16_constellation():
    """Generate 16-QAM constellation points."""
    points = {}
    idx = 0
    for i in [-3, -1, 1, 3]:
        for q in [-3, -1, 1, 3]:
            points[idx] = (i, q)
            idx += 1
    return points

def qam16_modulate(symbols, carrier_freq, samples_per_symbol, sample_rate):
    """16-QAM modulation."""
    const = qam16_constellation()
    result = []
    for sym in symbols:
        I, Q = const.get(sym % 16, (0, 0))
        for k in range(samples_per_symbol):
            t = k / sample_rate
            val = I * math.cos(TAU * carrier_freq * t) - Q * math.sin(TAU * carrier_freq * t)
            result.append(val)
    return result

def qam16_demodulate(signal, carrier_freq, samples_per_symbol, sample_rate):
    """16-QAM demodulation via correlation."""
    const = qam16_constellation()
    n_symbols = len(signal) // samples_per_symbol
    symbols = []
    for s in range(n_symbols):
        I_acc = 0.0
        Q_acc = 0.0
        for k in range(samples_per_symbol):
            t = k / sample_rate
            idx = s * samples_per_symbol + k
            I_acc += signal[idx] * math.cos(TAU * carrier_freq * t)
            Q_acc += -signal[idx] * math.sin(TAU * carrier_freq * t)
        I_acc *= 2.0 / samples_per_symbol
        Q_acc *= 2.0 / samples_per_symbol
        best_sym = 0
        best_dist = float('inf')
        for sym, (ci, cq) in const.items():
            d = (I_acc - ci)**2 + (Q_acc - cq)**2
            if d < best_dist:
                best_dist = d
                best_sym = sym
        symbols.append(best_sym)
    return symbols

def symbol_error_rate(tx_symbols, rx_symbols):
    """Compute symbol error rate."""
    n = min(len(tx_symbols), len(rx_symbols))
    if n == 0:
        return 0.0
    errors = sum(1 for i in range(n) if tx_symbols[i] != rx_symbols[i])
    return errors / n
