"""RQ-M03: fixed original samples, independent streams and endpoint stability.

Bootstrap prefixes share draws, not their bias corrections. Original-sample
properties (t control, standard error and acceleration) are computed once.
Evaluation metrics use original samples as the independent population unit;
the K inner streams measure conditional computational variation.
"""

from numbers import Integral, Real

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri

from mathscape.inference import t_inference

MAIN_NAMESPACE = 0x4D3033
SMOKE_NAMESPACE = 0x534D3033
DISTRIBUTION_IDS = {"normal": 1, "centered_exponential": 2}
METHODS = ("t", "percentile", "bca")
B_VALUES = (199, 999, 4999)
MAX_B = 4999
MAX_ELEMENTS = 1_000_000
MAX_N = 200
MAX_ID = 2**32 - 1


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _probability(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    return value


def _vector(values, label, minimum=1, maximum=None):
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        return None, f"{label}_not_one_dimensional_real"
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        return None, f"{label}_not_one_dimensional_real"
    if raw.size < minimum or (maximum is not None and raw.size > maximum):
        return None, f"{label}_size_out_of_range"
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        return array, f"{label}_nonfinite"
    return array, ""


def _sample(sample):
    sample, reason = _vector(sample, "sample", 2, MAX_N)
    if reason:
        return sample, reason
    if np.all(sample == sample[0]):
        return sample, "sample_constant"
    return sample, ""


def rng_for_panel(
    distribution, n, master_seed=0, outer_id=0, purpose_id=0, inner_id=0,
    namespace=MAIN_NAMESPACE,
):
    """PCG64 seed key: (namespace, distribution ID, n, outer, purpose, inner).

    Entropy is master_seed. IDs normal=1, centered_exponential=2;
    purpose original=0, bootstrap=1. An original stream always has inner=0.
    B, method, display order and batching never participate in stream selection.
    """
    if not isinstance(distribution, str) or distribution not in DISTRIBUTION_IDS:
        raise ValueError("distribution must be normal or centered_exponential")
    n = _integer(n, "n", 2, MAX_N)
    master_seed = _integer(master_seed, "master_seed", 0, MAX_ID)
    outer_id = _integer(outer_id, "outer_id", 0, 99)
    purpose_id = _integer(purpose_id, "purpose_id", 0, 1)
    inner_id = _integer(inner_id, "inner_id", 0, 19)
    namespace = _integer(namespace, "namespace", 0, MAX_ID)
    if purpose_id == 0 and inner_id != 0:
        raise ValueError("the original-sample stream must use inner_id=0")
    sequence = np.random.SeedSequence(
        entropy=master_seed,
        spawn_key=(namespace, DISTRIBUTION_IDS[distribution], n, outer_id, purpose_id, inner_id),
    )
    return np.random.Generator(np.random.PCG64(sequence))


def generate_sample(distribution, n, master_seed=0, outer_id=0, namespace=MAIN_NAMESPACE):
    rng = rng_for_panel(distribution, n, master_seed, outer_id, 0, 0, namespace)
    if distribution == "normal":
        return rng.standard_normal(n)
    return rng.exponential(scale=1.0, size=n) - 1.0


def bootstrap_means(sample, rng, Bmax=4999, max_elements=MAX_ELEMENTS):
    """Return one complete inner series with bounded resample matrix sizes."""
    sample, reason = _sample(sample)
    if reason:
        raise ValueError(reason)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    Bmax = _integer(Bmax, "Bmax", 1, MAX_B)
    max_elements = _integer(max_elements, "max_elements", sample.size, MAX_ELEMENTS)
    batch_size = max_elements // sample.size
    means = np.empty(Bmax)
    for start in range(0, Bmax, batch_size):
        count = min(batch_size, Bmax - start)
        indices = rng.integers(0, sample.size, size=(count, sample.size))
        with np.errstate(over="ignore", invalid="ignore"):
            means[start:start + count] = np.mean(sample[indices], axis=1)
    if not np.isfinite(means).all():
        raise ValueError("bootstrap_mean_nonfinite")
    return means


def generate_bootstrap_means(
    sample, distribution, n, master_seed=0, outer_id=0, inner_id=0, Bmax=4999,
    namespace=MAIN_NAMESPACE, max_elements=MAX_ELEMENTS,
):
    sample, reason = _sample(sample)
    if reason:
        raise ValueError(reason)
    if sample.size != n:
        raise ValueError("sample length and n disagree")
    rng = rng_for_panel(distribution, n, master_seed, outer_id, 1, inner_id, namespace)
    return bootstrap_means(sample, rng, Bmax, max_elements)


def _record(reason="", **details):
    return {
        "lower": float("nan"), "upper": float("nan"), "valid": False,
        "reason": reason, "z0": float("nan"), "a": float("nan"),
        "q_low": float("nan"), "q_high": float("nan"),
        "tau_low": float("nan"), "tau_high": float("nan"),
        "tail_warning": False, **details,
    }


def prepare_sample(sample, alpha=0.05):
    """Cache t and acceleration once from the original observations alone."""
    alpha = _probability(alpha, "alpha")
    sample, reason = _sample(sample)
    context = {"sample": sample, "alpha": alpha, "mean": float("nan"),
               "se": float("nan"), "a": float("nan"), "valid": False,
               "reason": reason, "t": _record(reason)}
    if reason:
        return context
    result = t_inference(sample, alpha=alpha)
    mean, se = float(result["mean"][0]), float(result["se"][0])
    if not np.isfinite(mean) or not np.isfinite(se) or se <= 0:
        context.update(mean=mean, se=se, reason="sample_se_nonpositive_or_nonfinite",
                       t=_record("sample_se_nonpositive_or_nonfinite"))
        return context
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        centered = sample - mean
        normalized = centered / np.max(np.abs(centered))
        acceleration = float(np.sum(normalized**3) / (6 * np.sum(normalized**2)**1.5))
    t = (_record(lower=float(result["lower"][0]), upper=float(result["upper"][0]),
                 valid=True, a=acceleration) if result["valid"][0]
         else _record("t_inference_undefined", a=acceleration))
    context.update(sample=sample.copy(), mean=mean, se=se, a=acceleration,
                   valid=True, reason="", t=t)
    return context


def _quantile_record(values, q_low, q_high, **details):
    ends = np.quantile(values, [q_low, q_high], method="linear")
    if not np.isfinite(ends).all() or ends[0] > ends[1]:
        return _record("interval_endpoints_nonfinite", q_low=float(q_low), q_high=float(q_high), **details)
    tau_low = float(values.size * min(q_low, 1 - q_low))
    tau_high = float(values.size * min(q_high, 1 - q_high))
    return _record(lower=float(ends[0]), upper=float(ends[1]), valid=True,
                   q_low=float(q_low), q_high=float(q_high), tau_low=tau_low,
                   tau_high=tau_high, tail_warning=min(tau_low, tau_high) < 10, **details)


def infer_prefixes(context, means, B_values=B_VALUES):
    """Infer every requested prefix without redrawing or reusing z0 across B.

    The context is made by prepare_sample. Its t record and a are unchanged
    across inner streams and prefixes. Only values within a requested prefix
    determine its validity: a nonfinite unused suffix does not invalidate an
    earlier finite prefix. Bca validity is the RQ-M02 convention:
    finite z0/a, positive adjustment denominators, and ordered probabilities
    0<=q_low<q_high<=1. No probabilities or interval endpoints are clipped.
    """
    if not isinstance(context, dict) or not {"valid", "reason", "alpha", "mean", "a", "t"} <= context.keys():
        raise ValueError("context must be created by prepare_sample")
    try:
        B_values = tuple(_integer(b, "B", 1, MAX_B) for b in B_values)
    except TypeError as exc:
        raise ValueError("B_values must be a nonempty sequence of integers") from exc
    if not B_values or len(set(B_values)) != len(B_values):
        raise ValueError("B_values must be nonempty and distinct")
    values, reason = _vector(means, "bootstrap_distribution", 1, MAX_B)
    if values is not None and values.size < max(B_values):
        raise ValueError("a requested prefix exceeds the available bootstrap distribution")
    if not context["valid"]:
        return {b: {m: _record(context["reason"]) for m in METHODS} for b in B_values}
    alpha, mean, acceleration = context["alpha"], context["mean"], context["a"]
    output = {}
    z_lower = ndtri(alpha / 2)
    z_bounds = np.array([z_lower, -z_lower])
    for b in B_values:
        prefix_reason = reason
        if reason == "bootstrap_distribution_nonfinite" and np.isfinite(values[:b]).all():
            prefix_reason = ""
        if prefix_reason:
            output[b] = {"t": context["t"].copy(), "percentile": _record(prefix_reason, a=acceleration),
                         "bca": _record(prefix_reason, a=acceleration)}
            continue
        prefix = values[:b]
        percentile = _quantile_record(prefix, alpha / 2, 1 - alpha / 2, a=acceleration)
        probability = (np.count_nonzero(prefix < mean) + .5 * np.count_nonzero(prefix == mean)) / b
        z0 = float(ndtri(probability))
        if not np.isfinite(z0):
            bca = _record("bca_bias_correction_nonfinite", z0=z0, a=acceleration)
        elif not np.isfinite(acceleration):
            bca = _record("bca_acceleration_undefined", z0=z0, a=acceleration)
        else:
            denominators = 1 - acceleration * (z0 + z_bounds)
            if not np.isfinite(denominators).all() or np.any(denominators <= 0):
                bca = _record("bca_adjustment_singular", z0=z0, a=acceleration)
            else:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    q_low, q_high = ndtr(z0 + (z0 + z_bounds) / denominators)
                if not np.isfinite([q_low, q_high]).all() or not 0 <= q_low < q_high <= 1:
                    bca = _record("bca_quantiles_invalid", z0=z0, a=acceleration,
                                  q_low=float(q_low), q_high=float(q_high))
                else:
                    bca = _quantile_record(prefix, q_low, q_high, z0=z0, a=acceleration)
        output[b] = {"t": context["t"].copy(), "percentile": percentile, "bca": bca}
    return output


def original_metrics(data, se):
    """Reduce one original sample's (B,method,K) endpoints to primary metrics.

    Required arrays are lower/upper/valid/tail_warning; method order is METHODS.
    t is a single deterministic control (effective K=1), never K independent
    observations. Any needed invalid inner stream makes all primary float
    metrics NA; counts remain. Bootstrap endpoint SDs use ddof=1. C is the
    mean of coverage indicators, not coverage of a newly averaged interval.
    """
    try:
        lower, upper = np.asarray(data["lower"], dtype=float), np.asarray(data["upper"], dtype=float)
        valid, warning = np.asarray(data["valid"]), np.asarray(data["tail_warning"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("data must contain lower/upper/valid/tail_warning arrays") from exc
    if lower.ndim != 3 or lower.shape[1] != len(METHODS) or lower.shape[2] < 1:
        raise ValueError("endpoint shape must be (B,3 methods,K)")
    if any(a.shape != lower.shape for a in (upper, valid, warning)) or valid.dtype.kind != "b" or warning.dtype.kind != "b":
        raise ValueError("endpoint and boolean mask shapes must match")
    shape = lower.shape[:2]
    output = {key: np.full(shape, np.nan) for key in (
        "lower_sd", "upper_sd", "A", "C", "mean_width", "miss_above", "miss_below", "warning_rate")}
    output.update(complete=np.zeros(shape, dtype=bool), valid_inner=np.zeros(shape, dtype=int),
                  invalid_inner=np.zeros(shape, dtype=int), total_inner=np.zeros(shape, dtype=int))
    try:
        scale_valid = not isinstance(se, (bool, np.bool_)) and isinstance(se, Real) and np.isfinite(se) and se > 0
    except (TypeError, ValueError):
        scale_valid = False
    for bi in range(shape[0]):
        for mi, method in enumerate(METHODS):
            if method == "t":
                for array in (lower, upper, valid, warning):
                    first = np.repeat(array[bi, mi, :1], lower.shape[2])
                    if not np.array_equal(array[bi, mi], first, equal_nan=True):
                        raise ValueError("t control must be invariant across inner streams")
            selection = slice(0, 1) if method == "t" else slice(None)
            lo, hi, mask = lower[bi, mi, selection], upper[bi, mi, selection], valid[bi, mi, selection]
            mask = mask & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
            total, count = mask.size, int(mask.sum())
            output["total_inner"][bi, mi] = total
            output["valid_inner"][bi, mi] = count
            output["invalid_inner"][bi, mi] = total - count
            complete = scale_valid and count == total and (method == "t" or total >= 2)
            output["complete"][bi, mi] = complete
            if not complete:
                continue
            lower_sd = 0.0 if method == "t" else float(np.std(lo, ddof=1))
            upper_sd = 0.0 if method == "t" else float(np.std(hi, ddof=1))
            if not np.isfinite([lower_sd, upper_sd, np.mean(hi - lo)]).all():
                output["complete"][bi, mi] = False
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                standardized = np.hypot(lower_sd, upper_sd) / np.sqrt(2) / se
            if not np.isfinite(standardized):
                output["complete"][bi, mi] = False
                continue
            output["lower_sd"][bi, mi] = lower_sd
            output["upper_sd"][bi, mi] = upper_sd
            output["A"][bi, mi] = standardized
            output["C"][bi, mi] = np.mean((lo <= 0) & (0 <= hi))
            output["mean_width"][bi, mi] = np.mean(hi - lo)
            output["miss_above"][bi, mi] = np.mean(lo > 0)
            output["miss_below"][bi, mi] = np.mean(hi < 0)
            output["warning_rate"][bi, mi] = np.mean(warning[bi, mi, selection])
    return output


def mean_t_interval(values, confidence=0.99):
    """Strict original-unit mean interval; retain N and never omit invalids.

    Constant values retain descriptive mean/SD/MCSE, but no confirmatory
    interval is defined, as preregistered. Bounds are never clipped to [0,1].
    """
    confidence = _probability(confidence, "confidence")
    array, reason = _vector(values, "values", minimum=0)
    n = 0 if array is None else array.size
    finite_count = 0 if array is None else int(np.isfinite(array).sum())
    result = {"N": n, "finite_count": finite_count, "mean": float("nan"),
              "sd": float("nan"), "mcse": float("nan"), "low": float("nan"),
              "high": float("nan"), "df": n - 1, "valid": False, "reason": reason}
    if reason or n == 0:
        result["reason"] = reason or "too_few_originals"
        return result
    if n > 1 and np.all(array == array[0]):
        return result | {"mean": float(array[0]), "sd": 0.0, "mcse": 0.0,
                         "reason": "variance_zero_or_nonfinite"}
    with np.errstate(over="ignore", invalid="ignore"):
        result["mean"] = float(np.mean(array))
    if not np.isfinite(result["mean"]):
        return result | {"mean": float("nan"), "reason": "statistic_nonfinite"}
    if n < 2:
        result["reason"] = "too_few_originals"
        return result
    sd = float(np.std(array, ddof=1))
    mcse = sd / np.sqrt(n)
    result.update(sd=sd, mcse=float(mcse))
    if not np.isfinite(sd) or sd <= 0:
        result["reason"] = "variance_zero_or_nonfinite"
        return result
    critical = stats.t.isf((1 - confidence) / 2, n - 1)
    low, high = result["mean"] - critical * mcse, result["mean"] + critical * mcse
    if not np.isfinite([low, high]).all():
        result["reason"] = "interval_nonfinite"
        return result
    return result | {"low": float(low), "high": float(high), "valid": True, "reason": ""}


def welch_interval(group1, group0, confidence=0.99, min_group=2):
    """Original-sample Welch interval, group1 minus group0, with size gate."""
    confidence = _probability(confidence, "confidence")
    min_group = _integer(min_group, "min_group", 2, 1000000)
    one, reason1 = _vector(group1, "group1", minimum=0)
    zero, reason0 = _vector(group0, "group0", minimum=0)
    n1, n0 = (0 if one is None else one.size), (0 if zero is None else zero.size)
    result = {"N1": n1, "N0": n0, "finite_count1": 0 if one is None else int(np.isfinite(one).sum()),
              "finite_count0": 0 if zero is None else int(np.isfinite(zero).sum()),
              **dict.fromkeys(("mean1", "mean0", "variance1", "variance0", "difference", "se", "df", "low", "high"), float("nan")),
              "valid": False, "reason": reason1 or reason0}
    if reason1 or reason0:
        return result
    if min(n1, n0) < min_group:
        return result | {"reason": "insufficient_group_count"}
    with np.errstate(over="ignore", invalid="ignore"):
        constant1, constant0 = np.all(one == one[0]), np.all(zero == zero[0])
        mean1 = float(one[0]) if constant1 else float(np.mean(one))
        mean0 = float(zero[0]) if constant0 else float(np.mean(zero))
        var1 = 0.0 if constant1 else float(np.var(one, ddof=1))
        var0 = 0.0 if constant0 else float(np.var(zero, ddof=1))
    if not np.isfinite([mean1, mean0, var1, var0]).all():
        return result | {"reason": "statistic_nonfinite"}
    difference = mean1 - mean0
    result.update(mean1=mean1, mean0=mean0, variance1=var1, variance0=var0, difference=difference)
    component1, component0 = var1 / n1, var0 / n0
    se2 = component1 + component0
    if not np.isfinite(se2) or se2 <= 0:
        return result | {"reason": "variance_zero_or_nonfinite"}
    # Normalize before squaring to avoid overflow in the Satterthwaite ratio.
    df = 1 / ((component1 / se2)**2 / (n1 - 1) + (component0 / se2)**2 / (n0 - 1))
    se = np.sqrt(se2)
    critical = stats.t.isf((1 - confidence) / 2, df)
    low, high = difference - critical * se, difference + critical * se
    if not np.isfinite([df, se, low, high]).all():
        return result | {"reason": "interval_nonfinite"}
    return result | {"se": float(se), "df": float(df), "low": float(low), "high": float(high), "valid": True, "reason": ""}


def quantity_status(low, high, threshold, direction, seed_points):
    """Strict interval/seed decision; clear opposite evidence has priority."""
    if direction not in ("less", "greater"):
        raise ValueError("direction must be less or greater")
    if not np.isfinite([low, high, threshold]).all() or low > high:
        return "inconclusive"
    contrary = low > threshold if direction == "less" else high < threshold
    if contrary:
        return "not_supported"
    if len(seed_points) != 3 or not np.isfinite(seed_points).all():
        return "inconclusive"
    desired = (high < threshold and all(p < threshold for p in seed_points)
               if direction == "less" else low > threshold and all(p > threshold for p in seed_points))
    return "supported" if desired else "inconclusive"
