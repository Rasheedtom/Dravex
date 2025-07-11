"""Adaptive threshold detection."""
import math

def fixed_threshold(x, threshold):
    """Detect samples above fixed threshold."""
    return [i for i, s in enumerate(x) if abs(s) > threshold]

def moving_average_threshold(x, window_size=100, multiplier=1.5):
    """Adaptive threshold based on local moving average."""
    n = len(x)
    detections = []
    for i in range(n):
        lo = max(0, i - window_size // 2)
        hi = min(n, i + window_size // 2)
        local_mean = sum(abs(x[j]) for j in range(lo, hi)) / (hi - lo)
        if abs(x[i]) > multiplier * local_mean:
            detections.append(i)
    return detections

def cfar_detector(x, guard_cells=2, reference_cells=8, threshold_factor=3.0):
    """Constant False Alarm Rate detector (radar-style)."""
    n = len(x)
    detections = []
    for i in range(reference_cells + guard_cells, n - reference_cells - guard_cells):
        leading = x[i - reference_cells - guard_cells:i - guard_cells]
        lagging = x[i + guard_cells + 1:i + guard_cells + reference_cells + 1]
        noise_est = (sum(abs(s) for s in leading) + sum(abs(s) for s in lagging)) / (2 * reference_cells)
        if abs(x[i]) > threshold_factor * noise_est:
            detections.append(i)
    return detections

def hysteresis_threshold(x, high_threshold, low_threshold):
    """Threshold with hysteresis (Schmitt trigger)."""
    state = False
    regions = []
    start = 0
    for i, s in enumerate(x):
        if not state and abs(s) > high_threshold:
            state = True
            start = i
        elif state and abs(s) < low_threshold:
            state = False
            regions.append((start, i))
    if state:
        regions.append((start, len(x)))
    return regions
