"""Kalman filter for signal smoothing and prediction.

Implements scalar and 2D Kalman filters for online
estimation of noisy signal values.
"""
from dravex.config import Config

class KalmanFilter:
    """1D scalar Kalman filter."""

    def __init__(self, process_noise=None, measurement_noise=None, initial_estimate=0.0):
        cfg = Config.get()
        self.q = process_noise if process_noise is not None else cfg.kalman_process_noise
        self.r = measurement_noise if measurement_noise is not None else cfg.kalman_measurement_noise
        self.x = initial_estimate
        self.p = 1.0

    def update(self, measurement):
        """Process a single measurement, return filtered estimate."""
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

    def filter_signal(self, measurements):
        """Filter an entire signal."""
        result = []
        for m in measurements:
            result.append(self.update(m))
        return result

    @property
    def gain(self):
        return self.p / (self.p + self.r)


class KalmanFilter2D:
    """2D Kalman filter (position + velocity)."""

    def __init__(self, dt=1.0, process_noise=1e-4, measurement_noise=0.1):
        self.dt = dt
        self.q = process_noise
        self.r = measurement_noise
        self.x = [0.0, 0.0]
        self.P = [[1.0, 0.0], [0.0, 1.0]]

    def predict(self):
        self.x[0] += self.dt * self.x[1]
        self.P[0][0] += self.dt * (self.P[1][0] + self.P[0][1]) + self.dt**2 * self.P[1][1] + self.q
        self.P[0][1] += self.dt * self.P[1][1]
        self.P[1][0] += self.dt * self.P[1][1]
        self.P[1][1] += self.q

    def update(self, measurement):
        self.predict()
        S = self.P[0][0] + self.r
        K0 = self.P[0][0] / S
        K1 = self.P[1][0] / S
        innovation = measurement - self.x[0]
        self.x[0] += K0 * innovation
        self.x[1] += K1 * innovation
        self.P[0][0] -= K0 * self.P[0][0]
        self.P[0][1] -= K0 * self.P[0][1]
        self.P[1][0] -= K1 * self.P[0][0]
        self.P[1][1] -= K1 * self.P[0][1]
        return self.x[0]

    def filter_signal(self, measurements):
        result = []
        for m in measurements:
            result.append(self.update(m))
        return result
