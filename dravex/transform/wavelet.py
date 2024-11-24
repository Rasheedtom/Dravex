"""Discrete Wavelet Transform (Haar and Daubechies-4)."""
import math

def haar_forward(x):
    """Haar wavelet forward transform.

    Uses the simplest wavelet basis: averages and differences.
    Input length should be a power of 2 for best results.
    """
    """Haar wavelet forward transform."""
    n = len(x)
    if n < 2:
        return list(x), []
    coeffs = list(x)
    details_all = []
    while n > 1:
        approx = []
        detail = []
        for i in range(0, n, 2):
            if i + 1 < n:
                approx.append((coeffs[i] + coeffs[i+1]) / math.sqrt(2))
                detail.append((coeffs[i] - coeffs[i+1]) / math.sqrt(2))
            else:
                approx.append(coeffs[i])
        details_all.append(detail)
        coeffs = approx
        n = len(coeffs)
    return coeffs, details_all

def haar_inverse(approx, details_all):
    """Haar wavelet inverse transform."""
    coeffs = list(approx)
    for detail in reversed(details_all):
        expanded = []
        for i in range(len(coeffs)):
            d = detail[i] if i < len(detail) else 0.0
            expanded.append((coeffs[i] + d) / math.sqrt(2))
            expanded.append((coeffs[i] - d) / math.sqrt(2))
        coeffs = expanded
    return coeffs

def db4_forward(x):
    """Daubechies-4 wavelet forward transform."""
    h = [
        (1 + math.sqrt(3)) / (4 * math.sqrt(2)),
        (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
        (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
        (1 - math.sqrt(3)) / (4 * math.sqrt(2)),
    ]
    g = [h[3], -h[2], h[1], -h[0]]
    n = len(x)
    approx = []
    detail = []
    for i in range(0, n - 3, 2):
        a = sum(h[k] * x[i + k] for k in range(4))
        d = sum(g[k] * x[i + k] for k in range(4))
        approx.append(a)
        detail.append(d)
    return approx, detail

def db4_inverse(approx, detail):
    """Daubechies-4 inverse transform."""
    h = [
        (1 + math.sqrt(3)) / (4 * math.sqrt(2)),
        (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
        (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
        (1 - math.sqrt(3)) / (4 * math.sqrt(2)),
    ]
    n = len(approx) * 2
    result = [0.0] * n
    for i in range(len(approx)):
        for k in range(4):
            idx = (2 * i + k) % n
            result[idx] += h[k] * approx[i] + (h[3-k] * ((-1)**k)) * detail[i]
    return result

def wavelet_denoise(x, threshold=0.1):
    """Denoise signal via wavelet thresholding."""
    approx, details = haar_forward(x)
    denoised_details = []
    for detail in details:
        denoised = [d if abs(d) > threshold else 0.0 for d in detail]
        denoised_details.append(denoised)
    return haar_inverse(approx, denoised_details)

def wavelet_packet(x, max_depth=3):
    """Wavelet packet decomposition (full binary tree of sub-bands)."""
    tree = {(0, 0): list(x)}
    for depth in range(max_depth):
        for node in range(2**depth):
            key = (depth, node)
            if key not in tree:
                continue
            data = tree[key]
            if len(data) < 2:
                continue
            approx = []
            detail = []
            for i in range(0, len(data) - 1, 2):
                approx.append((data[i] + data[i+1]) / 1.414)
                detail.append((data[i] - data[i+1]) / 1.414)
            tree[(depth+1, 2*node)] = approx
            tree[(depth+1, 2*node+1)] = detail
    return tree

def wavelet_energy_bands(x, n_levels=4):
    """Compute energy per wavelet sub-band."""
    approx = list(x)
    energies = []
    for _ in range(n_levels):
        if len(approx) < 2:
            break
        new_approx = []
        detail = []
        for i in range(0, len(approx) - 1, 2):
            new_approx.append((approx[i] + approx[i+1]) / 1.414)
            detail.append((approx[i] - approx[i+1]) / 1.414)
        energies.append(sum(d*d for d in detail))
        approx = new_approx
    energies.append(sum(a*a for a in approx))
    return list(reversed(energies))
