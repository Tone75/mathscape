"""Known dynamical examples and finite computation contracts, not chaos proofs."""

import numpy as np
import pytest

from mathscape.chaos import bifurcation, logistic_orbit, sensitivity


def test_orbit_is_deterministic_and_includes_initial_value():
    first = logistic_orbit(3.8, 0.2)
    np.testing.assert_array_equal(first, logistic_orbit(3.8, 0.2))
    assert first.shape == (81,)
    assert first[0] == 0.2
    assert first[1] == pytest.approx(3.8 * 0.2 * 0.8)
    np.testing.assert_array_equal(logistic_orbit(3.8, 0.2, 0), [0.2])


@pytest.mark.parametrize("r", [0.0, 1.0, 2.5, 3.2, 3.9, 4.0])
@pytest.mark.parametrize("x0", [0.0, 0.2, 0.5, 1.0])
def test_valid_orbits_remain_in_unit_interval(r, x0):
    values = logistic_orbit(r, x0, 500)
    assert np.isfinite(values).all()
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_stable_fixed_point_at_r_2_5():
    np.testing.assert_allclose(logistic_orbit(2.5, 0.2, 200)[-20:], 0.6, atol=1e-12)


def test_period_two_at_r_3_2():
    tail = logistic_orbit(3.2, 0.2, 200)[-20:]
    np.testing.assert_allclose(tail[2:], tail[:-2], atol=1e-12)
    assert abs(tail[-1] - tail[-2]) > 0.1
    expected = np.array([(4.2 - np.sqrt(0.2 * 4.2)) / 6.4,
                         (4.2 + np.sqrt(0.2 * 4.2)) / 6.4])
    np.testing.assert_allclose(np.sort(tail[-2:]), expected, atol=1e-12)


@pytest.mark.parametrize("x0", [0.0, 0.2, 1.0])
def test_sensitivity_returns_in_domain_orbits_and_difference(x0):
    first, second, difference = sensitivity(3.9, x0, 80, 1e-8)
    assert first.shape == second.shape == difference.shape == (81,)
    assert 0.0 <= second[0] <= 1.0
    assert second[0] != first[0]
    assert abs(second[0] - first[0]) == pytest.approx(1e-8, abs=1e-16)
    np.testing.assert_array_equal(first, logistic_orbit(3.9, x0, 80))
    np.testing.assert_array_equal(second, logistic_orbit(3.9, second[0], 80))
    np.testing.assert_array_equal(difference, np.abs(first - second))


def test_bifurcation_shape_finiteness_and_parameter_correspondence():
    r_values, values = bifurcation(samples=11, burn_in=40, keep=10)
    assert r_values.shape == (11,)
    assert values.shape == (10, 11)
    assert r_values[0] == 2.5
    assert r_values[-1] == 4.0
    assert np.isfinite(values).all()
    assert ((values >= 0.0) & (values <= 1.0)).all()
    for column, r in enumerate(r_values):
        np.testing.assert_array_equal(values[:, column], logistic_orbit(r, 0.2, 50)[41:])
    again = bifurcation(samples=11, burn_in=40, keep=10)
    np.testing.assert_array_equal(r_values, again[0])
    np.testing.assert_array_equal(values, again[1])


def test_bifurcation_without_burn_in():
    r_values, values = bifurcation(2.0, 3.0, samples=3, burn_in=0, keep=1)
    np.testing.assert_allclose(values[0], r_values * 0.2 * 0.8)


@pytest.mark.parametrize("kwargs", [
    {"r": -0.01}, {"r": 4.01}, {"r": np.nan}, {"r": np.inf}, {"r": True},
    {"x0": -0.01}, {"x0": 1.01}, {"x0": np.nan}, {"x0": "0.2"},
    {"n": -1}, {"n": 10_001}, {"n": 1.5}, {"n": True},
])
def test_invalid_orbit_parameters(kwargs):
    parameters = {"r": 3.8, "x0": 0.2, "n": 80} | kwargs
    with pytest.raises(ValueError):
        logistic_orbit(**parameters)


@pytest.mark.parametrize("kwargs", [
    {"delta": 0.0}, {"delta": -1e-8}, {"delta": np.nan}, {"delta": np.inf},
    {"delta": 1e-30}, {"delta": 0.9, "x0": 0.5}, {"r": 4.1}, {"n": -1},
])
def test_invalid_sensitivity_parameters(kwargs):
    parameters = {"r": 3.8, "x0": 0.2, "n": 80, "delta": 1e-8} | kwargs
    with pytest.raises(ValueError):
        sensitivity(**parameters)


@pytest.mark.parametrize("kwargs", [
    {"r_min": -1.0}, {"r_max": 4.1}, {"r_min": 4.0, "r_max": 2.5},
    {"r_min": np.nan}, {"r_max": np.inf}, {"x0": 1.1},
    {"samples": 1}, {"samples": 2_001}, {"samples": 10.5},
    {"burn_in": -1}, {"burn_in": 10_001}, {"keep": 0}, {"keep": 1_001},
    {"samples": 2_000, "burn_in": 2_000}, {"keep": True},
])
def test_invalid_bifurcation_parameters(kwargs):
    with pytest.raises(ValueError):
        bifurcation(**kwargs)
