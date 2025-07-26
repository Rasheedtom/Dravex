"""Distribution fitting for signal analysis."""
import math
from dravex.constants import DEFAULT_EPSILON

def fit_gaussian(x):
    """Fit Gaussian distribution to samples.

    Returns (mean, standard_deviation) tuple.
    """
    """Fit Gaussian distribution to samples."""
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    mu = sum(x) / n
    sigma = math.sqrt(sum((s - mu)**2 for s in x) / (n - 1))
    return mu, max(sigma, DEFAULT_EPSILON)

def gaussian_pdf(x, mu, sigma):
    """Evaluate Gaussian PDF."""
    if sigma < DEFAULT_EPSILON:
        return 1.0 if abs(x - mu) < DEFAULT_EPSILON else 0.0
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)

def fit_laplacian(x):
    """Fit Laplacian distribution (heavy tails)."""
    mu = sorted(x)[len(x) // 2]
    b = sum(abs(s - mu) for s in x) / max(1, len(x))
    return mu, max(b, DEFAULT_EPSILON)

def laplacian_pdf(x, mu, b):
    """Evaluate Laplacian PDF."""
    return math.exp(-abs(x - mu) / b) / (2 * b) if b > DEFAULT_EPSILON else 0.0

def kolmogorov_smirnov(x, cdf_func):
    """Kolmogorov-Smirnov test statistic."""
    sorted_x = sorted(x)
    n = len(sorted_x)
    max_d = 0.0
    for i, xi in enumerate(sorted_x):
        empirical = (i + 1) / n
        theoretical = cdf_func(xi)
        d = abs(empirical - theoretical)
        if d > max_d:
            max_d = d
    return max_d

def normal_cdf(x, mu=0, sigma=1):
    """Normal CDF using error function."""
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def fit_uniform(x):
    """Fit uniform distribution."""
    return min(x), max(x)

def uniform_pdf(x, a, b):
    """Uniform PDF."""
    if a >= b:
        return 0.0
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def chi_squared_test(observed, expected):
    """Chi-squared goodness of fit test statistic."""
    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi2 += (o - e)**2 / e
    return chi2

def anderson_darling(x, cdf_func):
    """Anderson-Darling test statistic."""
    import math
    sorted_x = sorted(x)
    n = len(sorted_x)
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(n):
        Fi = max(1e-12, min(1 - 1e-12, cdf_func(sorted_x[i])))
        Fn = max(1e-12, min(1 - 1e-12, cdf_func(sorted_x[n - 1 - i])))
        s += (2*i + 1) * (math.log(Fi) + math.log(1 - Fn))
    return -n - s / n

def jarque_bera(x):
    """Jarque-Bera test for normality."""
    from dravex.stats.moments import skewness, kurtosis
    n = len(x)
    if n < 3:
        return 0.0
    S = skewness(x)
    K = kurtosis(x)
    return n / 6 * (S**2 + K**2 / 4)

def empirical_cdf(x, value):
    """Evaluate empirical CDF at a value."""
    n = len(x)
    count = sum(1 for xi in x if xi <= value)
    return count / n
