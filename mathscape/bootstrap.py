"""Paired t, percentile and BCa mean intervals for the fixed RQ-M02 design.

The interval API receives observations and the bootstrap mean distribution,
never population truth or a distribution label. Percentile and BCa reuse the
same bootstrap means. No endpoint clipping, replacement method or removal of
undefined results is performed. Quantiles use NumPy's ``method='linear'``.
"""

from numbers import Integral, Real

import numpy as np
from scipy.special import ndtr, ndtri

from mathscape.inference import t_inference

MAIN_NAMESPACE = 0x4D3032
SMOKE_NAMESPACE = 0x534D3032
DISTRIBUTION_IDS = {"normal": 1, "centered_exponential": 2}
PURPOSE_IDS = {"original": 0, "bootstrap": 1}
METHODS = ("t", "percentile", "bca")
MAX_N = 200
MAX_BOOTSTRAP = 999
MAX_ELEMENTS = 1_000_000
MAX_ID = 2**32 - 1
TAIL_WARNING_COUNT = 10


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _alpha(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("alpha must be a finite real number strictly between zero and one")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("alpha must be finite") from exc
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("alpha must be strictly between zero and one")
    return value


def _vector(values, *, kind, min_size, max_size):
    """Return a numeric vector or a stable machine-readable invalid reason."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        return None, f"{kind}_not_one_dimensional_real"
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        return None, f"{kind}_not_one_dimensional_real"
    if not min_size <= raw.size <= max_size:
        return None, f"{kind}_size_out_of_range"
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        return None, f"{kind}_nonfinite"
    return array, ""


def _sample(values):
    sample, reason = _vector(values, kind="sample", min_size=2, max_size=MAX_N)
    if reason:
        return None, reason
    if np.all(sample == sample[0]):
        return None, "sample_constant"
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(sample)
    if not np.isfinite(mean):
        return None, "sample_statistic_nonfinite"
    return sample, ""


def _rng(namespace, distribution_id, n, replicate_id, purpose_id, master_seed):
    sequence = np.random.SeedSequence(
        entropy=master_seed,
        spawn_key=(namespace, distribution_id, n, replicate_id, purpose_id),
    )
    return np.random.Generator(np.random.PCG64(sequence))


def bootstrap_means(sample, rng, B_boot=999, max_elements=MAX_ELEMENTS):
    """Resample one sample with replacement and return B_boot sample means.

    The original sample is 1d, n=2..200, finite and nonconstant. B_boot is
    1..999. Each index matrix and resampled-value matrix has at most
    max_elements<=1,000,000 elements. Changing this batching cap preserves
    the random draws and means exactly; no shared index matrix spans outer
    replications. Invalid inputs raise ValueError before resampling.
    """
    sample, reason = _sample(sample)
    if reason:
        raise ValueError(reason)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    B_boot = _integer(B_boot, "B_boot", 1, MAX_BOOTSTRAP)
    max_elements = _integer(max_elements, "max_elements", sample.size, MAX_ELEMENTS)
    batch_size = max_elements // sample.size
    means = np.empty(B_boot, dtype=np.float64)
    for start in range(0, B_boot, batch_size):
        count = min(batch_size, B_boot - start)
        indices = rng.integers(0, sample.size, size=(count, sample.size))
        with np.errstate(over="ignore", invalid="ignore"):
            means[start:start + count] = np.mean(sample[indices], axis=1)
    if not np.all(np.isfinite(means)):
        raise ValueError("bootstrap_mean_nonfinite")
    return means


def draw_replicate(
    distribution, n, master_seed=0, replicate_id=0, B_boot=999,
    namespace=MAIN_NAMESPACE,
):
    """Return one original sample and its independent bootstrap mean draws.

    Both populations have mean 0 and variance 1: N(0,1), or Exp(scale=1)-1.
    RQ-M02's seed is PCG64(SeedSequence(entropy=master_seed,
    spawn_key=(namespace,distribution_id,n,replicate_id,purpose_id))).
    IDs: normal=1, centered_exponential=2; original=0, bootstrap=1.
    Method, alpha, execution order, B_boot and batch size are absent from the
    key. MAIN_NAMESPACE=0x4D3032, SMOKE_NAMESPACE=0x534D3032; neither reuses
    RQ-M01's mapping. Each outer replicate owns two distinct streams.
    """
    if not isinstance(distribution, str) or distribution not in DISTRIBUTION_IDS:
        raise ValueError("distribution must be normal or centered_exponential")
    n = _integer(n, "n", 2, MAX_N)
    master_seed = _integer(master_seed, "master_seed", 0, MAX_ID)
    replicate_id = _integer(replicate_id, "replicate_id", 0, MAX_ID)
    namespace = _integer(namespace, "namespace", 0, MAX_ID)
    B_boot = _integer(B_boot, "B_boot", 1, MAX_BOOTSTRAP)
    distribution_id = DISTRIBUTION_IDS[distribution]
    original_rng = _rng(namespace, distribution_id, n, replicate_id, 0, master_seed)
    resampling_rng = _rng(namespace, distribution_id, n, replicate_id, 1, master_seed)
    if distribution == "normal":
        sample = original_rng.standard_normal(n)
    else:
        sample = original_rng.exponential(scale=1.0, size=n) - 1.0
    return sample, bootstrap_means(sample, resampling_rng, B_boot)


def _record(reason="", **details):
    return {
        "lower": float("nan"), "upper": float("nan"), "valid": False,
        "reason": reason, "z0": float("nan"), "a": float("nan"),
        "q_low": float("nan"), "q_high": float("nan"),
        "tail_warning": False, "tail_count_low": float("nan"),
        "tail_count_high": float("nan"), **details,
    }


def _bootstrap_record(distribution, q_low, q_high, **details):
    lower, upper = np.quantile(distribution, [q_low, q_high], method="linear")
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
        return _record("interval_endpoints_nonfinite", q_low=q_low, q_high=q_high, **details)
    low_count = float(distribution.size * q_low)
    high_count = float(distribution.size * (1 - q_high))
    return _record(
        lower=float(lower), upper=float(upper), valid=True, reason="",
        q_low=float(q_low), q_high=float(q_high),
        tail_warning=bool(min(low_count, high_count) < TAIL_WARNING_COUNT),
        tail_count_low=low_count, tail_count_high=high_count, **details,
    )


def intervals(sample, bootstrap_distribution, alpha=0.05):
    """Return t/percentile/BCa records from one shared sample/distribution.

    Invalid original observations (including constants) invalidate all methods.
    An invalid bootstrap distribution invalidates only percentile/BCa. Each
    record has lower, upper, valid, reason, z0, a, q_low, q_high, tail_warning,
    tail_count_low, tail_count_high. Inapplicable diagnostics are NaN/False.
    Reasons are machine-readable; undefined intervals always retain NaN ends.

    BCa ties use (#less + .5*#equal)/B. Mean acceleration uses the centered
    sample cubic sum divided by six times its squared sum to the power 3/2;
    centered deviations are scaled first for numerical stability. Adjusted
    probabilities are unmodified. Nonfinite or nonpositive denominators and
    unordered/out-of-range probabilities invalidate BCa without fallback.
    Valid q=0 or q=1 is permitted by the linear empirical quantile definition
    and explicitly triggers a zero-tail-resolution warning. Finite tail
    counts below ten are warnings, not exclusions or reasons to increase B.
    """
    alpha = _alpha(alpha)
    sample, reason = _sample(sample)
    if reason:
        return {method: _record(reason) for method in METHODS}
    t_result = t_inference(sample, alpha=alpha)
    if t_result["valid"][0]:
        t_record = _record(lower=float(t_result["lower"][0]), upper=float(t_result["upper"][0]), valid=True)
    else:
        t_record = _record("t_inference_undefined")
    distribution, reason = _vector(
        bootstrap_distribution, kind="bootstrap_distribution",
        min_size=1, max_size=MAX_BOOTSTRAP,
    )
    if reason:
        return {"t": t_record, "percentile": _record(reason), "bca": _record(reason)}
    percentile = _bootstrap_record(distribution, alpha / 2, 1 - alpha / 2)
    mean = np.mean(sample)
    probability = (np.count_nonzero(distribution < mean) + 0.5 * np.count_nonzero(distribution == mean)) / distribution.size
    z0 = float(ndtri(probability))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        centered = sample - mean
        scale = np.max(np.abs(centered))
        normalized = centered / scale
        squared_sum = np.sum(normalized**2)
        acceleration = float(np.sum(normalized**3) / (6 * squared_sum**1.5))
    if not np.isfinite(z0):
        bca = _record("bca_bias_correction_nonfinite", z0=z0, a=acceleration)
    elif not np.isfinite(acceleration):
        bca = _record("bca_acceleration_undefined", z0=z0, a=acceleration)
    else:
        # ndtri(alpha/2) and its negation avoid subtractive cancellation at 1.
        z_lower = ndtri(alpha / 2)
        z_bounds = np.array([z_lower, -z_lower])
        denominators = 1 - acceleration * (z0 + z_bounds)
        if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0):
            bca = _record("bca_adjustment_singular", z0=z0, a=acceleration)
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                adjusted = ndtr(z0 + (z0 + z_bounds) / denominators)
            q_low, q_high = adjusted
            if not np.all(np.isfinite(adjusted)) or not 0 <= q_low < q_high <= 1:
                bca = _record("bca_quantiles_invalid", z0=z0, a=acceleration, q_low=float(q_low), q_high=float(q_high))
            else:
                bca = _bootstrap_record(distribution, q_low, q_high, z0=z0, a=acceleration)
    return {"t": t_record, "percentile": percentile, "bca": bca}
