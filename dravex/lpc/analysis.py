"""Linear Predictive Coding analysis."""
from dravex.constants import DEFAULT_EPSILON

def lpc_autocorrelation(x, order):
    """LPC analysis via autocorrelation + Levinson-Durbin."""
    n = len(x)
    r = [0.0] * (order + 1)
    for k in range(order + 1):
        for i in range(n - k):
            r[k] += x[i] * x[i + k]
    return levinson_durbin(r, order)

def levinson_durbin(r, order):
    """Levinson-Durbin recursion for LPC coefficients.

    Solves the Yule-Walker equations in O(p^2) time where p is the order.
    """
    """Levinson-Durbin recursion for LPC coefficients."""
    a = [0.0] * (order + 1)
    a[0] = 1.0
    e = r[0]
    if e < DEFAULT_EPSILON:
        return a
    for i in range(1, order + 1):
        lam = 0.0
        for j in range(1, i):
            lam += a[j] * r[i - j]
        lam = (r[i] - lam) / e if e > DEFAULT_EPSILON else 0.0
        lam = max(-0.999, min(0.999, lam))
        new_a = list(a)
        new_a[i] = lam
        for j in range(1, i):
            new_a[j] = a[j] - lam * a[i - j]
        a = new_a
        e *= (1 - lam * lam)
        if e < DEFAULT_EPSILON:
            break
    return a

def lpc_error(x, coeffs):
    """Compute LPC prediction error (residual)."""
    n = len(x)
    order = len(coeffs) - 1
    residual = [0.0] * n
    for i in range(n):
        prediction = 0.0
        for k in range(1, min(order + 1, i + 1)):
            prediction += coeffs[k] * x[i - k]
        residual[i] = x[i] + prediction
    return residual

def lpc_spectrum(coeffs, n_points=512):
    """Compute LPC spectrum (frequency response of all-pole model)."""
    import math
    from dravex.constants import TAU
    freqs = []
    magnitudes = []
    for i in range(n_points):
        w = TAU * i / (2 * n_points)
        freqs.append(w)
        re = sum(coeffs[k] * math.cos(-w * k) for k in range(len(coeffs)))
        im = sum(coeffs[k] * math.sin(-w * k) for k in range(len(coeffs)))
        mag = math.sqrt(re*re + im*im)
        if mag > DEFAULT_EPSILON:
            magnitudes.append(1.0 / mag)
        else:
            magnitudes.append(1e6)
    return freqs, magnitudes

def lpc_bandwidth(coeffs, sample_rate):
    """Estimate formant bandwidths from LPC poles."""
    import math
    from dravex.spectral.formant import _find_polynomial_roots
    roots = _find_polynomial_roots(coeffs)
    bandwidths = []
    for re, im in roots:
        if im > 1e-6:
            r = math.sqrt(re*re + im*im)
            bw = -math.log(r) * sample_rate / math.pi if r > 0 else 0
            bandwidths.append(abs(bw))
    return sorted(bandwidths)

def lpc_gain(x, coeffs):
    """Compute LPC prediction gain."""
    import math
    residual = lpc_error(x, coeffs)
    sig_power = sum(s*s for s in x) / max(1, len(x))
    res_power = sum(r*r for r in residual) / max(1, len(residual))
    if res_power < 1e-15:
        return 100.0
    return 10 * math.log10(sig_power / res_power)

def burg_method(x, order):
    """Burg's method for LPC analysis (better for short segments)."""
    n = len(x)
    a = [1.0] + [0.0] * order
    ef = list(x)
    eb = list(x)
    for i in range(1, order + 1):
        num = sum(ef[j]*eb[j-1] for j in range(i, n))
        den = sum(ef[j]*ef[j] + eb[j-1]*eb[j-1] for j in range(i, n))
        k = -2 * num / den if abs(den) > 1e-15 else 0.0
        new_a = list(a)
        for j in range(1, i):
            new_a[j] = a[j] + k * a[i - j]
        new_a[i] = k
        a = new_a
        new_ef = [0.0] * n
        new_eb = [0.0] * n
        for j in range(i, n):
            new_ef[j] = ef[j] + k * eb[j - 1]
            new_eb[j] = eb[j - 1] + k * ef[j]
        ef = new_ef
        eb = new_eb
    return a
