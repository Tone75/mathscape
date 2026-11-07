"""Deterministic finite-precision experiments with the logistic map.

These computations illustrate dynamical behaviour; a finite orbit is not a
proof of chaos or of an exact asymptotic bifurcation set.
"""

from numbers import Integral, Real

import numpy as np

MAX_ORBIT_STEPS = 10_000
MAX_BIFURCATION_SAMPLES = 2_000
MAX_BIFURCATION_UPDATES = 2_000_000


def _real(name: str, value: float, lower: float, upper: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    value = float(value)
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value


def _integer(name: str, value: int, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return int(value)


def logistic_orbit(r: float, x0: float, n: int = 80) -> np.ndarray:
    """Return x[0] through x[n], including the supplied initial condition.

    Zero iterations are supported. At most 10,000 iterations may be requested.
    Values use float64 arithmetic, without stochastic perturbations or clipping.
    """
    r = _real("r", r, 0.0, 4.0)
    x0 = _real("x0", x0, 0.0, 1.0)
    n = _integer("n", n, 0, MAX_ORBIT_STEPS)
    values = np.empty(n + 1, dtype=np.float64)
    values[0] = x0
    for index in range(n):
        values[index + 1] = r * values[index] * (1.0 - values[index])
    return values


def sensitivity(
    r: float, x0: float, n: int = 80, delta: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two nearby orbits and their pointwise absolute difference.

    Prefer x0 + delta; use x0 - delta when the forward perturbation would leave
    [0, 1]. Reject perturbations that cannot produce a distinct in-domain float.
    The actual initial difference is subject to floating-point rounding.
    """
    x0 = _real("x0", x0, 0.0, 1.0)
    delta = _real("delta", delta, 0.0, 1.0)
    if delta == 0.0:
        raise ValueError("delta must be positive.")
    nearby = x0 + delta if x0 + delta <= 1.0 else x0 - delta
    if not 0.0 <= nearby <= 1.0 or nearby == x0:
        raise ValueError("delta must produce a distinct initial value in [0, 1].")
    first = logistic_orbit(r, x0, n)
    second = logistic_orbit(r, nearby, n)
    return first, second, np.abs(first - second)


def bifurcation(
    r_min: float = 2.5,
    r_max: float = 4.0,
    samples: int = 400,
    burn_in: int = 400,
    keep: int = 80,
    x0: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample finite post-transient logistic trajectories, vectorized over r.

    Return (r_values, values), with shapes (samples,) and (keep, samples).
    The first retained row is x[burn_in + 1]. Each column corresponds to the
    same-index r value. Endpoints of the requested r interval are included.
    Work is bounded by 2,000,000 scalar map updates, 2,000 r values, and at most
    1,000 retained iterations, limiting both compute and output memory.
    """
    r_min = _real("r_min", r_min, 0.0, 4.0)
    r_max = _real("r_max", r_max, 0.0, 4.0)
    if r_min > r_max:
        raise ValueError("r_min must not exceed r_max.")
    x0 = _real("x0", x0, 0.0, 1.0)
    samples = _integer("samples", samples, 2, MAX_BIFURCATION_SAMPLES)
    burn_in = _integer("burn_in", burn_in, 0, MAX_ORBIT_STEPS)
    keep = _integer("keep", keep, 1, 1_000)
    if samples * (burn_in + keep) > MAX_BIFURCATION_UPDATES:
        raise ValueError("Requested bifurcation calculation exceeds the work limit.")
    r_values = np.linspace(r_min, r_max, samples, dtype=np.float64)
    state = np.full(samples, x0, dtype=np.float64)
    for _ in range(burn_in):
        state = r_values * state * (1.0 - state)
    values = np.empty((keep, samples), dtype=np.float64)
    for index in range(keep):
        state = r_values * state * (1.0 - state)
        values[index] = state
    return r_values, values
