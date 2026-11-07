"""Finite escape-time observations for the Mandelbrot iteration.

The count is the first n >= 1 for which |z_n| > 2, starting at z_0 = 0.
A zero count means escape was NOT observed before ``max_iter``; it does not
prove membership of the Mandelbrot set.
"""

from numbers import Integral, Real

import numpy as np

MAX_ITER = 150
MAX_RESOLUTION = 450
MIN_RESOLUTION = 32
MAX_PIXELS = MAX_RESOLUTION**2
MIN_WIDTH = 1e-6
MAX_WIDTH = 6.0


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def escape_counts(c, max_iter: int = 80):
    """Return first escape iterations, or 0 where escape is unconfirmed.

    Accept a numeric scalar or numeric array with at most 450**2 entries.
    Output has the input shape (Python ``int`` for a scalar). Already escaped
    points are excluded from all subsequent updates. ``max_iter`` is 1..150.
    """
    max_iter = _integer(max_iter, "max_iter", 1, MAX_ITER)
    try:
        raw = np.asarray(c)
    except (TypeError, ValueError) as exc:
        raise ValueError("c must contain finite numeric values") from exc
    if raw.size > MAX_PIXELS:
        raise ValueError(f"c must contain at most {MAX_PIXELS} points")
    if raw.dtype.kind not in "iufc":
        raise ValueError("c must contain finite numeric values")
    points = np.asarray(raw, dtype=np.complex128)
    if not np.all(np.isfinite(points)):
        raise ValueError("c must contain finite numeric values")
    flat = points.ravel()
    z = np.zeros(flat.size, dtype=np.complex128)
    counts = np.zeros(flat.size, dtype=np.uint16)
    active = np.arange(flat.size)
    for iteration in range(1, max_iter + 1):
        if not active.size:
            break
        # All active z values had |z| <= 2 at the previous observation.
        # A huge but finite c can have an infinite floating-point modulus;
        # it still unambiguously lies outside the escape radius.
        with np.errstate(over="ignore"):
            updated = z[active] ** 2 + flat[active]
            escaped = np.abs(updated) > 2.0
        z[active] = updated
        counts[active[escaped]] = iteration
        active = active[~escaped]
    result = counts.reshape(points.shape)
    return int(result.item()) if points.ndim == 0 else result


def mandelbrot(
    center_real: float = -0.7,
    center_imag: float = 0.0,
    width: float = 3.0,
    resolution: int = 360,
    max_iter: int = 80,
):
    """Return ``(real_axis, imaginary_axis, escape_counts)`` for a square.

    The count matrix is indexed [imaginary, real], both axes increase. Width
    is 1e-6..6, resolution is 32..450, and max_iter is 1..150. Axis bounds
    include the endpoints. Equal real/imaginary scales must be used in plots.
    The initial square includes the Mandelbrot set's complete extent.
    """
    center_real = _real(center_real, "center_real")
    center_imag = _real(center_imag, "center_imag")
    width = _real(width, "width")
    if not MIN_WIDTH <= width <= MAX_WIDTH:
        raise ValueError(f"width must be between {MIN_WIDTH} and {MAX_WIDTH}")
    resolution = _integer(resolution, "resolution", MIN_RESOLUTION, MAX_RESOLUTION)
    max_iter = _integer(max_iter, "max_iter", 1, MAX_ITER)
    half = width / 2
    x = np.linspace(center_real - half, center_real + half, resolution)
    y = np.linspace(center_imag - half, center_imag + half, resolution)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("view bounds must be finite")
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("view is too small to resolve at this center in float64")
    c = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    return x, y, escape_counts(c, max_iter)
