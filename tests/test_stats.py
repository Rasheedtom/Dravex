"""Tests for statistical analysis."""
import math
import random
from dravex.stats.distribution import fit_gaussian, gaussian_pdf, normal_cdf, kolmogorov_smirnov
from dravex.stats.entropy import shannon_entropy, sample_entropy, permutation_entropy
from dravex.stats.entropy import shannon_entropy, spectral_entropy
from dravex.stats.histogram import amplitude_histogram, cumulative_distribution, percentile, interquartile_range, dynamic_range
from dravex.stats.histogram import amplitude_histogram, percentile
from dravex.stats.moments import mean, variance, std_dev, skewness, kurtosis, median, mad, trimmed_mean, geometric_mean, harmonic_mean, peak_to_average_ratio, zero_crossing_rate
from dravex.stats.moments import mean, variance, std_dev, skewness, kurtosis, rms
from dravex.stats.periodicity import autocorrelation_periodicity, period_strength
from dravex.stats.periodicity import dominant_period_fft
from dravex.stats.periodicity import is_periodic
from dravex.stats.periodicity import period_jitter
from dravex.stats.stationarity import kpss_test_simple
from dravex.stats.stationarity import segment_stationarity
from dravex.stats.stationarity import trend_test
from dravex.stats.stationarity import variance_ratio, rolling_mean_test


class TestStats:
    def test_mean(self):
        assert abs(mean([1, 2, 3, 4, 5]) - 3.0) < 1e-10

    def test_variance(self):
        v = variance([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(v - 4.571) < 0.01

    def test_std_dev(self):
        assert abs(std_dev([1, 1, 1]) - 0.0) < 1e-10

    def test_skewness_symmetric(self):
        x = [-2, -1, 0, 1, 2]
        assert abs(skewness(x)) < 0.01

    def test_kurtosis_normal(self):
        rng = random.Random(42)
        x = [rng.gauss(0, 1) for _ in range(10000)]
        k = kurtosis(x)
        assert abs(k) < 0.3

    def test_rms(self):
        assert abs(rms([3, 4]) - math.sqrt(12.5)) < 1e-10

    def test_histogram(self):
        x = [0.0] * 50 + [1.0] * 50
        bins, counts = amplitude_histogram(x, 10)
        assert sum(counts) == 100

    def test_percentile(self):
        x = list(range(101))
        assert abs(percentile(x, 50) - 50) < 1e-6

    def test_entropy(self):
        x = [1.0] * 100
        e = shannon_entropy(x, 10)
        assert e == 0.0

    def test_spectral_entropy(self):
        uniform = [1.0] * 10
        e = spectral_entropy(uniform)
        assert e > 0



class TestStatsAdvanced:
    def test_median_odd(self):
        assert median([3, 1, 2]) == 2

    def test_median_even(self):
        assert abs(median([1, 2, 3, 4]) - 2.5) < 1e-10

    def test_mad(self):
        m = mad([1, 1, 2, 2, 4, 6, 9])
        assert m > 0

    def test_trimmed_mean(self):
        x = [0, 1, 1, 1, 1, 1, 1, 1, 1, 100]
        tm = trimmed_mean(x, 0.1)
        assert abs(tm - 1.0) < 1

    def test_geometric_mean(self):
        assert abs(geometric_mean([4, 9]) - 6.0) < 1e-6

    def test_harmonic_mean(self):
        assert abs(harmonic_mean([1, 4]) - 1.6) < 1e-6

    def test_papr(self):
        x = [0, 0, 0, 10, 0, 0]
        papr = peak_to_average_ratio(x)
        assert papr > 1

    def test_zcr(self):
        x = [1, -1, 1, -1]
        assert abs(zero_crossing_rate(x) - 1.0) < 1e-10

    def test_cumulative(self):
        x = list(range(100))
        bins, cdf = cumulative_distribution(x, 10)
        assert abs(cdf[-1] - 1.0) < 1e-10

    def test_iqr(self):
        x = list(range(100))
        iqr = interquartile_range(x)
        assert 40 < iqr < 60

    def test_dynamic_range(self):
        x = [0.1] * 50 + [1.0] * 50
        dr = dynamic_range(x)
        assert dr > 0

    def test_fit_gaussian(self):
        rng = random.Random(42)
        x = [rng.gauss(5, 2) for _ in range(1000)]
        mu, sigma = fit_gaussian(x)
        assert abs(mu - 5) < 0.5
        assert abs(sigma - 2) < 0.5

    def test_gaussian_pdf(self):
        p = gaussian_pdf(0, 0, 1)
        assert abs(p - 0.3989) < 0.001

    def test_normal_cdf(self):
        assert abs(normal_cdf(0, 0, 1) - 0.5) < 1e-6

    def test_ks_test(self):
        x = list(range(100))
        x = [xi / 100 for xi in x]
        stat = kolmogorov_smirnov(x, lambda v: v)
        assert stat < 0.02

    def test_sample_entropy(self):
        x = [math.sin(0.1*i) for i in range(200)]
        se = sample_entropy(x, 2)
        assert se >= 0

    def test_permutation_entropy(self):
        x = list(range(100))
        pe = permutation_entropy(x, 3)
        assert 0 <= pe <= 1

    def test_variance_ratio(self):
        x = [1.0] * 1024
        vr = variance_ratio(x, 256)
        assert vr >= 0

    def test_rolling_mean_stationary(self):
        x = [1.0 + 0.01 * (i % 10) for i in range(1000)]
        assert rolling_mean_test(x, 100, 0.5)

    def test_periodicity(self):
        x = [math.sin(2*math.pi*i/50) for i in range(500)]
        period = autocorrelation_periodicity(x, 10, 100)
        assert 45 < period < 55

    def test_period_strength(self):
        x = [math.sin(2*math.pi*i/20) for i in range(200)]
        ps = period_strength(x, 20)
        assert ps > 0.5


class TestStationarityAdvanced:
    def test_kpss(self):
        x = [1.0 + 0.01 * (i % 5) for i in range(200)]
        stat, is_stationary = kpss_test_simple(x)
        assert isinstance(is_stationary, bool)

    def test_segment_stationarity(self):
        x = [1.0] * 400
        assert segment_stationarity(x, 4)

    def test_trend_test_no_trend(self):
        x = [1.0] * 100
        has_trend, r = trend_test(x)
        assert not has_trend

    def test_trend_test_with_trend(self):
        x = [float(i) for i in range(100)]
        has_trend, r = trend_test(x)
        assert has_trend
        assert r > 0.9


class TestPeriodicityExtra:
    def test_dominant_period_fft(self):
        x = [math.sin(2*math.pi*i/50) for i in range(500)]
        period = dominant_period_fft(x, 500)
        assert 40 < period < 60

    def test_is_periodic(self):
        periodic = [math.sin(2*math.pi*i/20) for i in range(200)]
        assert is_periodic(periodic)

    def test_period_jitter(self):
        x = [math.sin(2*math.pi*i/50) for i in range(1000)]
        j = period_jitter(x, 50)
        assert j < 0.1

    def test_not_periodic(self):
        rng = random.Random(42)
        noise = [rng.gauss(0, 1) for _ in range(500)]
        assert not is_periodic(noise)