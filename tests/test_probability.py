"""Mathematical checks use full supports, separately from display windows."""

import numpy as np
import pytest
from scipy.integrate import quad

from mathscape.probability import (
    beta_moments,
    beta_pdf,
    gamma_moments,
    gamma_pdf,
    normal_moments,
    normal_pdf,
    probability_curve,
)


def test_normal_known_values_and_moments():
    assert normal_pdf(0) == pytest.approx(1 / np.sqrt(2 * np.pi))
    assert normal_pdf(1) == pytest.approx(np.exp(-0.5) / np.sqrt(2 * np.pi))
    assert normal_pdf(2, mu=2, sigma=3) == pytest.approx(1 / (3 * np.sqrt(2 * np.pi)))
    assert normal_moments(2, 3) == pytest.approx((2, 9))


def test_gamma_exponential_and_moments():
    x = np.array([0, 0.1, 1, 3, 10])
    np.testing.assert_allclose(gamma_pdf(x, k=1, theta=2), np.exp(-x / 2) / 2)
    assert gamma_moments(3, 2) == pytest.approx((6, 12))
    assert gamma_pdf(-1, k=0.5, theta=2) == 0


def test_beta_uniform_and_moments():
    np.testing.assert_allclose(beta_pdf(np.linspace(0, 1, 11), 1, 1), 1)
    assert beta_moments(2, 3) == pytest.approx((2 / 5, 1 / 25))
    np.testing.assert_array_equal(beta_pdf([-1, 2], 0.5, 0.5), [0, 0])


@pytest.mark.parametrize(
    "pdf, params, lower, upper",
    [
        (normal_pdf, {"mu": 0, "sigma": 1}, -np.inf, np.inf),
        (normal_pdf, {"mu": -3, "sigma": 0.2}, -np.inf, np.inf),
        (gamma_pdf, {"k": 0.5, "theta": 0.2}, 0, np.inf),
        (gamma_pdf, {"k": 8, "theta": 5}, 0, np.inf),
        (beta_pdf, {"alpha": 0.5, "beta": 0.5}, 0, 1),
        (beta_pdf, {"alpha": 10, "beta": 0.5}, 0, 1),
        (beta_pdf, {"alpha": 2, "beta": 3}, 0, 1),
    ],
)
def test_normalization_on_full_support(pdf, params, lower, upper):
    value, error = quad(lambda x: pdf(x, **params), lower, upper, epsabs=1e-9, limit=200)
    assert value == pytest.approx(1, abs=2e-8)
    assert error < 2e-8


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf, -np.inf, True, 1j, "1", [1]])
@pytest.mark.parametrize(
    "function, key",
    [
        (normal_pdf, "sigma"),
        (gamma_pdf, "k"),
        (gamma_pdf, "theta"),
        (beta_pdf, "alpha"),
        (beta_pdf, "beta"),
    ],
)
def test_pdf_rejects_invalid_positive_parameters(function, key, invalid):
    with pytest.raises(ValueError):
        function(0.5, **{key: invalid})


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, True, 1j, "1", [1]])
def test_normal_rejects_invalid_location(invalid):
    with pytest.raises(ValueError):
        normal_pdf(0, mu=invalid)


@pytest.mark.parametrize(
    "function, params",
    [(normal_moments, {"sigma": 0}), (gamma_moments, {"theta": np.nan}), (beta_moments, {"alpha": -1})],
)
def test_moments_validate_parameters(function, params):
    with pytest.raises(ValueError):
        function(**params)


def test_singular_endpoint_densities_are_preserved():
    assert np.isposinf(gamma_pdf(0, k=0.5))
    assert gamma_pdf(0, k=1, theta=2) == pytest.approx(0.5)
    assert gamma_pdf(0, k=2) == 0
    assert np.isposinf(beta_pdf(0, alpha=0.5, beta=2))
    assert np.isposinf(beta_pdf(1, alpha=2, beta=0.5))
    assert np.isposinf(beta_pdf([0, 1], 0.5, 0.5)).all()


@pytest.mark.parametrize(
    "distribution, params, pdf",
    [
        ("normal", {"mu": -5, "sigma": 0.2}, normal_pdf),
        ("gamma", {"k": 0.5, "theta": 5}, gamma_pdf),
        ("gamma", {"k": 8, "theta": 0.2}, gamma_pdf),
        ("beta", {"alpha": 0.5, "beta": 0.5}, beta_pdf),
        ("beta", {"alpha": 10, "beta": 0.5}, beta_pdf),
    ],
)
def test_finite_display_grid_uses_actual_density(distribution, params, pdf):
    x, y = probability_curve(distribution, points=800, **params)
    assert x.shape == y.shape == (800,)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert (np.diff(x) > 0).all()
    np.testing.assert_allclose(y, pdf(x, **params))
    if distribution == "gamma":
        assert (x > 0).all()
    if distribution == "beta":
        assert ((0 < x) & (x < 1)).all()


def test_display_window_is_not_the_full_normalization_domain():
    x, _ = probability_curve("normal")
    mass, _ = quad(normal_pdf, x[0], x[-1])
    assert mass == pytest.approx(0.999)
    assert mass < 1


@pytest.mark.parametrize("points", [0, 49, 5001, 100.5, True])
def test_display_grid_size_validation(points):
    with pytest.raises(ValueError):
        probability_curve("normal", points=points)


def test_unknown_distribution_and_parameters_rejected():
    with pytest.raises(ValueError):
        probability_curve("unknown")
    with pytest.raises(ValueError):
        probability_curve("gamma", rate=2)


def test_unrepresentable_display_grid_rejected():
    with pytest.raises(ValueError):
        probability_curve("normal", mu=1e100, sigma=1e-100)
