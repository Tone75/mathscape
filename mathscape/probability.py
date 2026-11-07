"""Probability densities, moments and finite grids independent of the UI.

Gamma uses ``shape=k, scale=theta, loc=0`` throughout. PDF functions preserve
infinite endpoint densities; only the display grids omit singular endpoints.
"""

from numbers import Integral, Real

import numpy as np
from scipy import stats


def _finite_real(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}")
    return value


def _normal(mu: float, sigma: float):
    mu = _finite_real(mu, "mu")
    sigma = _finite_real(sigma, "sigma", positive=True)
    return stats.norm(loc=mu, scale=sigma)


def _gamma(k: float, theta: float):
    k = _finite_real(k, "k", positive=True)
    theta = _finite_real(theta, "theta", positive=True)
    return stats.gamma(a=k, loc=0, scale=theta)


def _beta(alpha: float, beta: float):
    alpha = _finite_real(alpha, "alpha", positive=True)
    beta = _finite_real(beta, "beta", positive=True)
    return stats.beta(a=alpha, b=beta, loc=0, scale=1)


def normal_pdf(x, mu: float = 0.0, sigma: float = 1.0):
    """Evaluate the normal PDF (sigma is standard deviation, not variance)."""
    return _normal(mu, sigma).pdf(x)


def gamma_pdf(x, k: float = 2.0, theta: float = 1.0):
    """Evaluate the gamma PDF, including the right-limit value at zero.

    In particular, the endpoint density is +inf for 0 < k < 1, and the
    density is zero outside the nonnegative support.
    """
    return _gamma(k, theta).pdf(x)


def beta_pdf(x, alpha: float = 2.0, beta: float = 2.0):
    """Evaluate the beta PDF, preserving infinite densities at 0 or 1."""
    return _beta(alpha, beta).pdf(x)


def normal_moments(mu: float = 0.0, sigma: float = 1.0) -> tuple[float, float]:
    """Return (mean, variance)."""
    dist = _normal(mu, sigma)
    return float(dist.mean()), float(dist.var())


def gamma_moments(k: float = 2.0, theta: float = 1.0) -> tuple[float, float]:
    """Return (k * theta, k * theta**2) for shape/scale parameters."""
    dist = _gamma(k, theta)
    return float(dist.mean()), float(dist.var())


def beta_moments(alpha: float = 2.0, beta: float = 2.0) -> tuple[float, float]:
    """Return (mean, variance) on [0, 1]."""
    dist = _beta(alpha, beta)
    return float(dist.mean()), float(dist.var())


def probability_curve(
    distribution: str, *, points: int = 800, **params
) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite *display* grid and its unmodified PDF values.

    Normal: quantiles 0.0005 through 0.9995, uniformly spaced in x.
    Gamma: positive uniform grid up to quantile 0.999, with x=0 omitted.
    Beta: uniform interior grid in (0, 1), with both endpoints omitted.

    These are plotting windows, not normalization domains. Endpoint omission
    prevents infinity from being plotted; it does not redefine a density or
    imply that its limiting value is zero. Extreme parameters which cannot
    produce a representable finite grid raise ValueError explicitly.
    """
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, Integral):
        raise ValueError("points must be an integer between 50 and 5000")
    if not 50 <= points <= 5000:
        raise ValueError("points must be an integer between 50 and 5000")
    factories = {"normal": _normal, "gamma": _gamma, "beta": _beta}
    if distribution not in factories:
        raise ValueError("distribution must be normal, gamma or beta")
    defaults = {
        "normal": {"mu": 0.0, "sigma": 1.0},
        "gamma": {"k": 2.0, "theta": 1.0},
        "beta": {"alpha": 2.0, "beta": 2.0},
    }
    if params.keys() - defaults[distribution].keys():
        raise ValueError("Unexpected distribution parameter")
    dist = factories[distribution](**(defaults[distribution] | params))
    if distribution == "normal":
        lower, upper = dist.ppf([0.0005, 0.9995])
        x = np.linspace(lower, upper, points)
    elif distribution == "gamma":
        upper = dist.ppf(0.999)
        x = np.linspace(0.0, upper, points + 1)[1:]
    else:
        x = np.linspace(0.0, 1.0, points + 2)[1:-1]
    y = np.asarray(dist.pdf(x), dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Parameters do not produce a finite display grid")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Parameters exceed the precision of the display grid")
    return x, y
