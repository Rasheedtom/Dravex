"""Tests for Kalman filter."""
from dravex.filter.kalman import KalmanFilter, KalmanFilter2D

class TestKalman:
    def test_converges(self):
        kf = KalmanFilter(process_noise=0.01, measurement_noise=1.0)
        measurements = [5.0 + 0.1*i for i in range(50)]
        filtered = kf.filter_signal(measurements)
        assert abs(filtered[-1] - measurements[-1]) < 2.0

    def test_smooth(self):
        kf = KalmanFilter(process_noise=0.001, measurement_noise=0.5)
        noisy = [1.0, 1.5, 0.8, 1.2, 0.9]
        filtered = kf.filter_signal(noisy)
        assert len(filtered) == 5

    def test_2d(self):
        kf = KalmanFilter2D(dt=1.0)
        measurements = [float(i) for i in range(20)]
        filtered = kf.filter_signal(measurements)
        assert abs(filtered[-1] - measurements[-1]) < 5
