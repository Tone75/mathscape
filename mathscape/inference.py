"""Independent simulation and one-sample, two-sided unknown-variance t inference.

Only the generator knows population truth. ``t_inference`` accepts observations,
the null mean and alpha; it never receives true sigma or the true mean. All
sample variances use ddof=1 and all t distributions use df=n-1.

Inference is undefined for a nonfinite sample, zero sample variance or a
nonrepresentable intermediate result. Such rows remain present, have valid=False
and NaN inferential results, and are never silently counted as nonrejections.
If even one row is invalid, ``summarize`` reports undefined (NaN) rates and width
aggregates while preserving the original denominator and observed valid counts.

Decisions use unrounded p < alpha. Confidence intervals are closed intervals.
For numerical CI/test correspondence checks only, observations whose |t| is
within atol=1e-12 + rtol=1e-12*critical of the critical value are classified as
boundary observations; no p value, endpoint or actual decision is adjusted.
"""

from numbers import Integral, Real
import struct

import numpy as np
from scipy import stats

MAX_N = 200
MAX_REPETITIONS = 5_000
MAX_OBSERVATIONS = 1_000_000
MAX_SEED = 2**32 - 1
DISTRIBUTION_IDS = {"normal": 1, "centered_exponential": 2}
BOUNDARY_ATOL = 1e-12
BOUNDARY_RTOL = 1e-12


def _real(value, name, *, low=None, high=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _alpha(value):
    value = _real(value, "alpha")
    if not 0 < value < 1:
        raise ValueError("alpha must be strictly between zero and one")
    return value


def _distribution(value):
    if not isinstance(value, str) or value not in DISTRIBUTION_IDS:
        raise ValueError("distribution must be normal or centered_exponential")
    return value


def _shape(n, repetitions):
    n = _integer(n, "n", 2, MAX_N)
    repetitions = _integer(repetitions, "repetitions", 1, MAX_REPETITIONS)
    if n * repetitions > MAX_OBSERVATIONS:
        raise ValueError("n * repetitions must not exceed 1000000")
    return n, repetitions


def rng_for_cell(distribution, n, d, master_seed, trial_id=0):
    """Return a PCG64 stream determined by explicit, order-independent IDs.

    Distribution IDs are normal=1, centered_exponential=2. After canonicalizing
    signed zero, d's IEEE754 binary64 bits are split into low/high uint32 words.
    SeedSequence entropy is master_seed, and spawn_key is
    (distribution_id, n, d_low32, d_high32, trial_id). No Python hash, alpha,
    repetition count or execution-order counter participates in stream selection.
    """
    distribution = _distribution(distribution)
    n = _integer(n, "n", 2, MAX_N)
    d = _real(d, "d", low=-1, high=1)
    master_seed = _integer(master_seed, "master_seed", 0, MAX_SEED)
    trial_id = _integer(trial_id, "trial_id", 0, MAX_SEED)
    d = 0.0 if d == 0 else d
    bits = struct.unpack("<Q", struct.pack("<d", d))[0]
    key = (DISTRIBUTION_IDS[distribution], n, bits & 0xFFFFFFFF, bits >> 32, trial_id)
    sequence = np.random.SeedSequence(entropy=master_seed, spawn_key=key)
    return np.random.Generator(np.random.PCG64(sequence))


def sample_batch(rng, distribution, n, repetitions, d=0, mu0=0, sigma=1):
    """Generate a row-major (repetitions,n) batch of independent observations.

    X=mu0+sigma*(d+Z), where Z is N(0,1) or Exp(scale=1)-1. Both models
    therefore have population mean mu0+sigma*d and population variance sigma².
    Sequential batches consume the same stream as one equally sized batch.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    distribution = _distribution(distribution)
    n, repetitions = _shape(n, repetitions)
    d = _real(d, "d", low=-1, high=1)
    mu0 = _real(mu0, "mu0")
    sigma = _real(sigma, "sigma", positive=True)
    if distribution == "normal":
        standardized = rng.standard_normal((repetitions, n))
    else:
        standardized = rng.exponential(scale=1.0, size=(repetitions, n)) - 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        samples = mu0 + sigma * (d + standardized)
    if not np.all(np.isfinite(samples)):
        raise ValueError("generation parameters exceed finite float64 range")
    return samples


def generate_samples(
    distribution, n, repetitions=500, d=0, seed=0, mu0=0, sigma=1, trial_id=0
):
    """Generate reproducible samples; alpha is deliberately absent from the API."""
    rng = rng_for_cell(distribution, n, d, seed, trial_id)
    return sample_batch(rng, distribution, n, repetitions, d, mu0, sigma)


def t_inference(samples, mu0=0, alpha=0.05):
    """Return row-wise t inference; a one-dimensional sample yields one row.

    Keys mean, variance, sd, se, t, p, lower, upper are float arrays; valid and
    reject are boolean arrays. Undefined inferential quantities are NaN, not
    substituted p=0/1. The null is rejected iff the unrounded two-sided p<alpha.
    """
    mu0 = _real(mu0, "mu0")
    alpha = _alpha(alpha)
    try:
        raw = np.asarray(samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples must be a rectangular real numeric array") from exc
    if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2):
        raise ValueError("samples must be a one- or two-dimensional real numeric array")
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    repetitions, n = raw.shape
    _shape(n, repetitions)
    values = np.asarray(raw, dtype=np.float64)
    finite_rows = np.all(np.isfinite(values), axis=1)
    means = np.full(repetitions, np.nan)
    variances = np.full(repetitions, np.nan)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        means[finite_rows] = np.mean(values[finite_rows], axis=1)
        variances[finite_rows] = np.var(values[finite_rows], axis=1, ddof=1)
        sd = np.sqrt(variances)
        se = sd / np.sqrt(n)
    valid = finite_rows & np.isfinite(means) & np.isfinite(variances) & (variances > 0) & (se > 0)
    t_values = np.full(repetitions, np.nan)
    p_values = np.full(repetitions, np.nan)
    lower = np.full(repetitions, np.nan)
    upper = np.full(repetitions, np.nan)
    critical = stats.t.isf(alpha / 2, n - 1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t_values[valid] = (means[valid] - mu0) / se[valid]
        p_values[valid] = 2 * stats.t.sf(np.abs(t_values[valid]), n - 1)
        lower[valid] = means[valid] - critical * se[valid]
        upper[valid] = means[valid] + critical * se[valid]
    valid &= np.isfinite(t_values) & np.isfinite(p_values) & np.isfinite(lower) & np.isfinite(upper)
    for array in (t_values, p_values, lower, upper):
        array[~valid] = np.nan
    return {
        "mean": means, "variance": variances, "sd": sd, "se": se,
        "t": t_values, "p": p_values, "lower": lower, "upper": upper,
        "valid": valid, "reject": valid & (p_values < alpha),
    }


def wilson_interval(count, total, confidence=0.95):
    """Return a two-sided Wilson score interval, including zero/all events."""
    total = _integer(total, "total", 1, 2**53)
    count = _integer(count, "count", 0, total)
    confidence = _real(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    z = stats.norm.isf((1 - confidence) / 2)
    rate = count / total
    z2_over_n = z * z / total
    denominator = 1 + z2_over_n
    center = (rate + z2_over_n / 2) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / total + z2_over_n / (4 * total)) / denominator
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


def summarize(result, true_mean):
    """Aggregate all rows; coverage always refers to the supplied true mean.

    No invalid row is removed from the denominator. If any row is invalid,
    reject_rate, coverage, all width aggregates and uncertainty summaries are
    undefined. Counts then describe observed valid rows only and must not be
    divided by the denominator as if missing outcomes were non-events.
    Width SD uses ddof=1; its MCSE is SD/sqrt(B). Binomial rate MCSE is
    sqrt(rate*(1-rate)/B), accompanied by a 95% Wilson interval even at 0/B.
    """
    true_mean = _real(true_mean, "true_mean")
    try:
        valid = np.asarray(result["valid"])
        reject = np.asarray(result["reject"])
        lower = np.asarray(result["lower"], dtype=float)
        upper = np.asarray(result["upper"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("result must contain inference arrays") from exc
    if valid.ndim != 1 or valid.dtype.kind != "b" or reject.dtype.kind != "b":
        raise ValueError("valid and reject must be one-dimensional boolean arrays")
    if not 1 <= valid.size <= MAX_REPETITIONS:
        raise ValueError("result has an invalid repetition count")
    if any(array.shape != valid.shape for array in (reject, lower, upper)):
        raise ValueError("result arrays must have matching one-dimensional shapes")
    if np.any(valid & (~np.isfinite(lower) | ~np.isfinite(upper) | (lower > upper))):
        raise ValueError("valid confidence intervals must have finite ordered endpoints")
    total = int(valid.size)
    invalid_count = int(np.count_nonzero(~valid))
    reject_count = int(np.count_nonzero(valid & reject))
    cover_count = int(np.count_nonzero(valid & (lower <= true_mean) & (true_mean <= upper)))
    output = {
        "denominator": total, "invalid_count": invalid_count,
        "reject_count": reject_count, "cover_count": cover_count,
    }
    metric_names = (
        "reject_rate", "coverage", "mean_width", "width_sd", "width_mcse",
        "reject_mcse", "coverage_mcse", "reject_low", "reject_high", "coverage_low", "coverage_high",
    )
    if invalid_count:
        return output | dict.fromkeys(metric_names, float("nan"))
    widths = upper - lower
    width_sd = float(np.std(widths, ddof=1)) if total > 1 else float("nan")
    reject_rate = reject_count / total
    coverage = cover_count / total
    reject_low, reject_high = wilson_interval(reject_count, total)
    coverage_low, coverage_high = wilson_interval(cover_count, total)
    return output | {
        "reject_rate": reject_rate, "coverage": coverage,
        "mean_width": float(np.mean(widths)), "width_sd": width_sd,
        "width_mcse": width_sd / np.sqrt(total),
        "reject_mcse": float(np.sqrt(reject_rate * (1 - reject_rate) / total)),
        "coverage_mcse": float(np.sqrt(coverage * (1 - coverage) / total)),
        "reject_low": reject_low, "reject_high": reject_high,
        "coverage_low": coverage_low, "coverage_high": coverage_high,
    }


def normal_power(n, d, alpha=0.05):
    """Exact two-sided t-test power for iid normal observations only.

    The noncentrality is sqrt(n)*d, where d=(true_mean-mu0)/sigma. This is a
    model-based probability over repeated samples, not power from one sample.
    """
    n = _integer(n, "n", 2, MAX_N)
    d = _real(d, "d", low=-1, high=1)
    alpha = _alpha(alpha)
    critical = stats.t.isf(alpha / 2, n - 1)
    noncentrality = np.sqrt(n) * d
    power = stats.nct.cdf(-critical, n - 1, noncentrality) + stats.nct.sf(critical, n - 1, noncentrality)
    if not np.isfinite(power):
        raise ValueError("normal power is not representable for these parameters")
    return float(power)
