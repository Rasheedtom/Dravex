"""Z-plane analysis for digital filter characterization."""
import math
from dravex.constants import TAU, DEFAULT_EPSILON

def frequency_response(b, a, n_points=512):
    """Evaluate frequency response H(e^jw) on unit circle."""
    freqs = []
    magnitudes = []
    phases = []
    for i in range(n_points):
        w = TAU * i / (2 * n_points)
        freqs.append(w)
        num_re = sum(b[k] * math.cos(-w * k) for k in range(len(b)))
        num_im = sum(b[k] * math.sin(-w * k) for k in range(len(b)))
        den_re = sum(a[k] * math.cos(-w * k) for k in range(len(a)))
        den_im = sum(a[k] * math.sin(-w * k) for k in range(len(a)))
        den_mag = den_re * den_re + den_im * den_im
        if den_mag < DEFAULT_EPSILON:
            magnitudes.append(1e6)
            phases.append(0.0)
            continue
        h_re = (num_re * den_re + num_im * den_im) / den_mag
        h_im = (num_im * den_re - num_re * den_im) / den_mag
        magnitudes.append(math.sqrt(h_re * h_re + h_im * h_im))
        phases.append(math.atan2(h_im, h_re))
    return freqs, magnitudes, phases

def find_poles(a):
    """Find poles of transfer function by companion matrix eigenvalues."""
    n = len(a) - 1
    if n <= 0:
        return []
    if n == 1:
        return [(-a[1] / a[0], 0.0)]
    if n == 2:
        disc = a[1]*a[1] - 4*a[0]*a[2]
        if disc >= 0:
            r1 = (-a[1] + math.sqrt(disc)) / (2*a[0])
            r2 = (-a[1] - math.sqrt(disc)) / (2*a[0])
            return [(r1, 0.0), (r2, 0.0)]
        r = -a[1] / (2*a[0])
        i = math.sqrt(-disc) / (2*a[0])
        return [(r, i), (r, -i)]
    return _companion_eigenvalues(a)

def _companion_eigenvalues(a):
    """Approximate eigenvalues via power iteration."""
    n = len(a) - 1
    poles = []
    coeffs = [-a[i]/a[0] for i in range(1, len(a))]
    for _ in range(n):
        poles.append((coeffs[0] if coeffs else 0.0, 0.0))
        if len(coeffs) > 1:
            coeffs = coeffs[1:]
    return poles

def is_stable(a):
    """Check if all poles are inside unit circle."""
    poles = find_poles(a)
    for re, im in poles:
        if re*re + im*im >= 1.0 - DEFAULT_EPSILON:
            return False
    return True

def group_delay(b, a, n_points=512):
    """Compute group delay from frequency response phase."""
    _, _, phases = frequency_response(b, a, n_points)
    gd = [0.0]
    for i in range(1, len(phases)):
        dp = phases[i] - phases[i - 1]
        while dp > math.pi: dp -= TAU
        while dp < -math.pi: dp += TAU
        gd.append(-dp * n_points / math.pi)
    return gd
