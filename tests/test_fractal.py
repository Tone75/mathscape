"""Check finite escape observations against an independent scalar loop."""

import numpy as np
import pytest

from mathscape.fractal import escape_counts, mandelbrot


def scalar_reference(c, max_iter):
    z = 0j
    for n in range(1, max_iter + 1):
        z = z * z + c
        if abs(z) > 2:
            return n
    return 0


@pytest.mark.parametrize(
    "c,expected", [(0, 0), (-1, 0), (1, 3), (2, 2), (3, 1), (-2, 0), (2j, 2)]
)
def test_known_escape_counts(c, expected):
    assert escape_counts(c, 80) == expected


def test_strict_escape_radius_and_iteration_cutoff():
    assert escape_counts(2, 1) == 0  # |z_1| == 2 is not escape.
    assert escape_counts(2, 2) == 2
    assert escape_counts(1, 2) == 0
    assert escape_counts(1, 3) == 3


def test_vector_counts_match_scalar_reference():
    x = np.linspace(-2.1, 0.9, 37)
    y = np.linspace(-1.3, 1.3, 31)
    points = x[None, :] + 1j * y[:, None]
    expected = np.array([[scalar_reference(c, 95) for c in row] for row in points])
    np.testing.assert_array_equal(escape_counts(points, 95), expected)


def test_previous_escape_observations_never_change():
    points = np.linspace(-2, 1, 400) + 0.2j
    short = escape_counts(points, 20)
    long = escape_counts(points, 150)
    observed = short > 0
    np.testing.assert_array_equal(short[observed], long[observed])
    assert np.all((long[~observed] == 0) | (long[~observed] > 20))


def test_grid_axes_shape_and_parameter_correspondence():
    x, y, counts = mandelbrot(-0.7, 0.2, 3.0, 32, 50)
    assert counts.shape == (32, 32)
    assert x.shape == y.shape == (32,)
    assert x[[0, -1]].tolist() == pytest.approx([-2.2, 0.8])
    assert y[[0, -1]].tolist() == pytest.approx([-1.3, 1.7])
    np.testing.assert_array_equal(counts, escape_counts(x[None, :] + 1j * y[:, None], 50))
    assert np.isfinite(counts).all()
    assert counts.min() >= 0 and counts.max() <= 50


def test_empty_and_large_finite_inputs():
    assert escape_counts(np.array([], dtype=complex)).shape == (0,)
    assert escape_counts(1e308 + 1e308j) == 1


@pytest.mark.parametrize("value", [0, -1, 151, 1.5, True, np.nan, np.inf])
def test_iteration_limits(value):
    with pytest.raises(ValueError):
        escape_counts(0, value)
    with pytest.raises(ValueError):
        mandelbrot(max_iter=value)


@pytest.mark.parametrize("value", [0, 31, 451, 32.5, True, np.nan, np.inf])
def test_resolution_limits(value):
    with pytest.raises(ValueError):
        mandelbrot(resolution=value)


def test_pixel_cap():
    with pytest.raises(ValueError, match="at most"):
        escape_counts(np.zeros(450 * 450 + 1))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, complex(0, np.nan), "0", True])
def test_nonfinite_or_nonnumeric_points_rejected(value):
    with pytest.raises(ValueError):
        escape_counts(value)


@pytest.mark.parametrize("name", ["center_real", "center_imag", "width"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "1"])
def test_invalid_view_values(name, value):
    with pytest.raises(ValueError):
        mandelbrot(**{name: value})


@pytest.mark.parametrize("width", [0, -1, 1e-7, 6.1])
def test_width_limits(width):
    with pytest.raises(ValueError):
        mandelbrot(width=width)


def test_unresolvable_view_rejected():
    with pytest.raises(ValueError, match="resolve"):
        mandelbrot(center_real=1e100, resolution=32)


def test_upper_limits_and_determinism():
    first = mandelbrot(center_real=3, resolution=450, max_iter=150)
    second = mandelbrot(center_real=3, resolution=450, max_iter=150)
    assert first[2].shape == (450, 450)
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)
