"""Formant detection using LPC analysis."""
import math
from dravex.constants import PI, DEFAULT_EPSILON

def find_formants(signal_samples, sample_rate, n_formants=4, lpc_order=None):
    """Detect formant frequencies using LPC pole analysis.

    Formants are resonant frequencies of the vocal tract,
    identified as poles of the LPC all-pole model near the unit circle.
    """
    """Detect formant frequencies using LPC pole analysis."""
    if lpc_order is None:
        lpc_order = 2 * n_formants + 2
    from dravex.lpc.analysis import lpc_autocorrelation
    coeffs = lpc_autocorrelation(signal_samples, lpc_order)
    roots = _find_polynomial_roots(coeffs)
    formants = []
    for re, im in roots:
        if im > DEFAULT_EPSILON:
            freq = math.atan2(im, re) * sample_rate / (2 * PI)
            bw = -math.log(math.sqrt(re*re + im*im) + DEFAULT_EPSILON) * sample_rate / PI
            if 90 < freq < sample_rate / 2 - 50 and bw < 400:
                formants.append((freq, bw))
    formants.sort(key=lambda f: f[0])
    return formants[:n_formants]

def _find_polynomial_roots(coeffs):
    """Find roots of polynomial using Durand-Kerner method."""
    n = len(coeffs) - 1
    if n <= 0:
        return []
    roots = []
    for k in range(n):
        angle = 2 * PI * k / n + 0.4
        r = 0.9 * math.cos(angle)
        i = 0.9 * math.sin(angle)
        roots.append((r, i))
    for _ in range(100):
        new_roots = []
        for i in range(n):
            pr, pi = _eval_poly(coeffs, roots[i])
            denom_re, denom_im = 1.0, 0.0
            for j in range(n):
                if j != i:
                    dr = roots[i][0] - roots[j][0]
                    di = roots[i][1] - roots[j][1]
                    nr = denom_re * dr - denom_im * di
                    ni = denom_re * di + denom_im * dr
                    denom_re, denom_im = nr, ni
            mag = denom_re*denom_re + denom_im*denom_im
            if mag > DEFAULT_EPSILON:
                qr = (pr*denom_re + pi*denom_im) / mag
                qi = (pi*denom_re - pr*denom_im) / mag
                new_roots.append((roots[i][0] - qr, roots[i][1] - qi))
            else:
                new_roots.append(roots[i])
        roots = new_roots
    return roots

def _eval_poly(coeffs, z):
    """Evaluate polynomial at complex point z."""
    re, im = 0.0, 0.0
    zr, zi = 1.0, 0.0
    for c in coeffs:
        re += c * zr
        im += c * zi
        new_zr = zr * z[0] - zi * z[1]
        new_zi = zr * z[1] + zi * z[0]
        zr, zi = new_zr, new_zi
    return re, im

def formant_track(signal_samples, sample_rate, frame_size=512, hop_size=256, n_formants=3):
    """Track formants across frames."""
    n = len(signal_samples)
    formant_tracks = [[] for _ in range(n_formants)]
    pos = 0
    while pos + frame_size <= n:
        frame = signal_samples[pos:pos + frame_size]
        formants = find_formants(frame, sample_rate, n_formants)
        for i in range(n_formants):
            if i < len(formants):
                formant_tracks[i].append(formants[i][0])
            else:
                formant_tracks[i].append(0.0)
        pos += hop_size
    return formant_tracks

def vowel_space(f1, f2):
    """Map F1-F2 to vowel space quadrant."""
    if f1 < 400:
        if f2 > 1500:
            return "front-close"
        elif f2 < 800:
            return "back-close"
        else:
            return "central-close"
    elif f1 > 600:
        if f2 > 1500:
            return "front-open"
        elif f2 < 800:
            return "back-open"
        else:
            return "central-open"
    else:
        return "mid"
