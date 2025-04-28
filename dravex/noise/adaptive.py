"""Adaptive noise cancellation using LMS algorithm."""
import math

def lms_filter(desired, reference, order=32, step_size=0.01):
    """LMS adaptive noise cancellation.

    Step size controls convergence speed vs stability tradeoff.
    Smaller values give more stable but slower convergence.
    """
    """LMS adaptive noise cancellation."""
    n = len(desired)
    w = [0.0] * order
    output = [0.0] * n
    error = [0.0] * n
    for i in range(order, n):
        ref_vec = reference[i-order:i][::-1]
        est = sum(wj * rj for wj, rj in zip(w, ref_vec))
        error[i] = desired[i] - est
        output[i] = error[i]
        for j in range(order):
            w[j] += step_size * error[i] * ref_vec[j]
    return output, error, w

def nlms_filter(desired, reference, order=32, step_size=0.5, eps=1e-6):
    """Normalized LMS adaptive filter."""
    n = len(desired)
    w = [0.0] * order
    output = [0.0] * n
    for i in range(order, n):
        ref_vec = reference[i-order:i][::-1]
        est = sum(wj * rj for wj, rj in zip(w, ref_vec))
        error = desired[i] - est
        output[i] = error
        power = sum(r * r for r in ref_vec) + eps
        mu = step_size / power
        for j in range(order):
            w[j] += mu * error * ref_vec[j]
    return output, w

def rls_filter(desired, reference, order=16, forget_factor=0.99, delta=1.0):
    """Recursive Least Squares adaptive filter."""
    n = len(desired)
    w = [0.0] * order
    P = [[delta if i == j else 0.0 for j in range(order)] for i in range(order)]
    output = [0.0] * n
    lam = forget_factor
    for i in range(order, n):
        x = reference[i-order:i][::-1]
        y_hat = sum(wj * xj for wj, xj in zip(w, x))
        error = desired[i] - y_hat
        output[i] = error
        Px = [sum(P[j][k] * x[k] for k in range(order)) for j in range(order)]
        denom = lam + sum(x[k] * Px[k] for k in range(order))
        k_gain = [px / denom for px in Px]
        for j in range(order):
            w[j] += k_gain[j] * error
        for j in range(order):
            for k in range(order):
                P[j][k] = (P[j][k] - k_gain[j] * Px[k]) / lam
    return output, w
