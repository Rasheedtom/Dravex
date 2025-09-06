"""Partial correlation (PARCOR) coefficients.

PARCOR coefficients are bounded to [-1, 1] and represent
the reflection coefficients of the lattice filter structure.
"""
import math
from dravex.constants import DEFAULT_EPSILON

def lpc_to_parcor(lpc_coeffs):
    """Convert LPC coefficients to PARCOR (reflection) coefficients."""
    order = len(lpc_coeffs) - 1
    a = list(lpc_coeffs)
    parcor = [0.0] * order
    for i in range(order, 0, -1):
        parcor[i - 1] = a[i]
        k = a[i]
        if abs(1 - k * k) < DEFAULT_EPSILON:
            break
        new_a = [0.0] * (i + 1)
        new_a[0] = 1.0
        for j in range(1, i):
            new_a[j] = (a[j] + k * a[i - j]) / (1 - k * k)
        a = new_a
    return parcor

def parcor_to_lpc(parcor):
    """Convert PARCOR coefficients back to LPC."""
    order = len(parcor)
    a = [1.0] + [0.0] * order
    for i in range(order):
        k = parcor[i]
        new_a = list(a)
        new_a[i + 1] = k
        for j in range(1, i + 1):
            new_a[j] = a[j] - k * a[i + 1 - j]
        a = new_a
    return a

def lar_from_parcor(parcor):
    """Log Area Ratios from PARCOR coefficients."""
    lar = []
    for k in parcor:
        k_clamp = max(-0.9999, min(0.9999, k))
        lar.append(math.log((1 + k_clamp) / (1 - k_clamp)))
    return lar

def parcor_from_lar(lar):
    """PARCOR coefficients from Log Area Ratios."""
    return [math.tanh(l / 2) for l in lar]

def lattice_filter(x, parcor):
    """Apply lattice filter using PARCOR coefficients."""
    order = len(parcor)
    n = len(x)
    f = [0.0] * (order + 1)
    b = [0.0] * (order + 1)
    output = [0.0] * n
    for i in range(n):
        f[0] = x[i]
        b[0] = x[i]
        for k in range(order):
            f_new = f[k] + parcor[k] * b[k]
            b_new = parcor[k] * f[k] + b[k]
            f[k + 1] = f_new
            b[k + 1] = b_new
        output[i] = f[order]
        for k in range(order, 0, -1):
            b[k] = b[k - 1]
    return output
