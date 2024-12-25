"""Moving average filters."""
import math

def simple_moving_average(x, window_size):
    """Simple moving average filter."""
    n = len(x)
    result = []
    for i in range(n):
        lo = max(0, i - window_size + 1)
        result.append(sum(x[lo:i+1]) / (i + 1 - lo))
    return result

def exponential_moving_average(x, alpha=0.1):
    """Exponential moving average (EMA)."""
    result = [x[0]]
    for i in range(1, len(x)):
        result.append(alpha * x[i] + (1 - alpha) * result[-1])
    return result

def weighted_moving_average(x, weights):
    """Weighted moving average."""
    n = len(x)
    m = len(weights)
    w_sum = sum(weights)
    result = []
    for i in range(n):
        s = 0.0
        w_total = 0.0
        for j in range(m):
            idx = i - m + 1 + j
            if 0 <= idx < n:
                s += x[idx] * weights[j]
                w_total += weights[j]
        result.append(s / w_total if w_total > 0 else 0.0)
    return result

def double_ema(x, alpha=0.1):
    """Double exponential smoothing (Holt's method)."""
    if not x:
        return []
    level = x[0]
    trend = x[1] - x[0] if len(x) > 1 else 0
    result = [level]
    for i in range(1, len(x)):
        new_level = alpha * x[i] + (1 - alpha) * (level + trend)
        new_trend = alpha * (new_level - level) + (1 - alpha) * trend
        level = new_level
        trend = new_trend
        result.append(level + trend)
    return result
