"""Independent saved-input audit for RQ-M04; no primary model/runner imports.

No observation or bootstrap generation occurs. Endpoint summaries, original-unit
MCSE, paired descriptive differences and independent-population hypotheses are
reconstructed separately. Bootstrap arrays are not saved: actual prefix reuse,
rank counts and interpolation cannot be empirically reconstructed by this audit.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import special, stats

METHODS = ("t", "percentile", "bca")
BS = (999, 4999)
DISTRIBUTIONS = ("normal", "gamma_0.5", "gamma_1", "gamma_2", "gamma_4", "gamma_16")
MAIN_NAMESPACE = 0x4D3034
METRICS = ("lower_sd", "upper_sd", "S", "Ase", "Apop", "C", "Mbelow", "Mabove", "T", "mean_width", "warning_rate")

def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def sample_hash(sample):
    return hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes(order="C")).hexdigest()


def _sd(values):
    values = np.asarray(values, dtype=float)
    return 0.0 if np.all(values == values[0]) else float(np.std(values, ddof=1))


def _original_se(samples):
    """Reconstruct early sample rejection before a rounded variance is used.

    Exactly constant fractional values can produce tiny positive np.std values.
    Such samples were rejected before inference, so their saved SE is undefined.
    """
    samples = np.asarray(samples, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        se = np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
        rejected = (~np.isfinite(samples).all(axis=1) | np.all(samples == samples[:, :1], axis=1)
                    | ~np.isfinite(np.mean(samples, axis=1)))
    se[rejected] = np.nan
    return se


def mean_interval(values, confidence=0.99):
    """Independent-original mean +/- Student critical value; no clipping."""
    values = np.asarray(values, dtype=float)
    total = values.size
    output = dict(N=total, finite_count=int(np.isfinite(values).sum()), mean=None, sd=None,
                  mcse=None, low=None, high=None, df=total - 1, defined=False)
    if total == 0 or not np.all(np.isfinite(values)):
        return output
    output["mean"] = float(np.mean(values))
    if total < 2:
        return output
    mean, sd = float(np.mean(values)), _sd(values)
    output.update(mean=mean, sd=sd, mcse=sd / np.sqrt(total))
    if sd <= 0:
        return output
    half = stats.t.ppf((1 + confidence) / 2, total - 1) * output["mcse"]
    return output | dict(low=float(mean - half), high=float(mean + half), defined=True)


def welch_interval(first, second, confidence=0.99, *, min_group=2):
    """Difference first-minus-second at the independent original-sample level."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    output = dict(N1=first.size, N0=second.size, finite_count1=int(np.isfinite(first).sum()),
                  finite_count0=int(np.isfinite(second).sum()), mean1=None, mean0=None, variance1=None, variance0=None,
                  difference=None, mcse=None, df=None,
                  low=None, high=None, defined=False)
    if min(first.size, second.size) < max(2, min_group) or not (np.isfinite(first).all() and np.isfinite(second).all()):
        return output
    difference = float(np.mean(first) - np.mean(second))
    first_var, second_var = _sd(first) ** 2, _sd(second) ** 2
    output.update(mean1=float(np.mean(first)), mean0=float(np.mean(second)),
                  variance1=float(first_var), variance0=float(second_var))
    parts = np.array([first_var / first.size, second_var / second.size])
    variance = float(parts.sum())
    denominator = parts[0] ** 2 / (first.size - 1) + parts[1] ** 2 / (second.size - 1)
    output["difference"] = difference
    if variance <= 0 or denominator <= 0:
        return output
    mcse, df = np.sqrt(variance), variance**2 / denominator
    half = stats.t.ppf((1 + confidence) / 2, df) * mcse
    return output | dict(mcse=float(mcse), df=float(df), low=float(difference - half),
                         high=float(difference + half), defined=True)


def _meta(panel, name):
    value = np.asarray(panel[name])
    if value.ndim != 0:
        raise ValueError(f"metadata {name} must be scalar")
    return value.item()


def inspect_panel(panel, *, strict_design=True):
    """Return diagnostic mismatches, without using generated random sequences."""
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    samples = np.asarray(panel["samples"], dtype=float)
    r, n = samples.shape
    b_values, methods, inner_ids = np.asarray(panel["B_values"]), list(panel["methods"]), np.asarray(panel["inner_id"])
    k = inner_ids.size
    shape = (r, len(b_values), len(methods), k)
    check(methods == list(METHODS), "method order is not t/percentile/bca")
    check(np.array_equal(panel["outer_id"], np.arange(r)), "outer IDs are not complete ordered prefixes")
    check(np.array_equal(inner_ids, np.arange(k)), "inner IDs are not common ordered k prefixes")
    check(_meta(panel, "n") == n and _meta(panel, "R") == r and _meta(panel, "K") == k, "metadata/sample dimensions disagree")
    check(_meta(panel, "alpha") == 0.05, "fixed alpha differs from 0.05")
    if strict_design:
        check(r == 400 and n == 20 and k == 10, "main R400/n20/K10 differs from fixed design")
        check(tuple(b_values) == BS, "main B prefixes differ from fixed design")
        check(_meta(panel, "namespace") == MAIN_NAMESPACE, "main namespace differs from fixed design")
        check(_meta(panel, "distribution") in DISTRIBUTIONS, "unknown population")
        check(_meta(panel, "master_seed") in (0, 1, 2), "unknown master seed")
    fields = ("lower", "upper", "valid", "reason", "tail_warning", "z0", "a", "q_low", "q_high", "tau_low", "tau_high")
    for name in fields:
        check(np.asarray(panel[name]).shape == shape, f"{name} shape differs from common R/B/method/k grid")
    if errors:
        return errors
    lower, upper, valid, reason = (np.asarray(panel[name]) for name in ("lower", "upper", "valid", "reason"))
    check(valid.dtype.kind == "b", "valid mask is not boolean")
    check(np.asarray(panel["tail_warning"]).dtype.kind == "b", "tail-warning mask is not boolean")
    check(np.all(np.isfinite(lower[valid])) and np.all(np.isfinite(upper[valid])), "valid endpoint is nonfinite")
    check(np.all(lower[valid] <= upper[valid]), "valid endpoints are reversed")
    check(np.all(np.isnan(lower[~valid])) and np.all(np.isnan(upper[~valid])), "invalid endpoint has a substituted value")
    check(np.all(reason[valid] == "") and np.all(reason[~valid] != ""), "valid/reason contract violated")
    hashes = np.asarray(panel["sample_sha256"])
    check(hashes.shape == (r,) and all(hashes[i] == sample_hash(sample) for i, sample in enumerate(samples)), "original sample hash mismatch")
    se = _original_se(samples)
    original_valid = np.isfinite(samples).all(axis=1) & np.isfinite(se) & (se > 0)
    if "se" in panel:
        check(np.allclose(panel["se"], se, rtol=1e-12, atol=1e-14, equal_nan=True), "saved se differs from originals")
    critical = stats.t.ppf(0.975, n - 1)
    for index in range(r):
        check(np.all(valid[index, :, 0] == original_valid[index]), f"outer {index}: t validity differs from original")
        if original_valid[index]:
            expected = (np.mean(samples[index]) - critical * se[index], np.mean(samples[index]) + critical * se[index])
            check(np.allclose(lower[index, :, 0], expected[0], atol=1e-12, rtol=1e-12)
                  and np.allclose(upper[index, :, 0], expected[1], atol=1e-12, rtol=1e-12), f"outer {index}: t reference mismatch")
            check(np.all(lower[index, :, 0] == lower[index, 0, 0, 0])
                  and np.all(upper[index, :, 0] == upper[index, 0, 0, 0]), f"outer {index}: t changes with B/k")
            jackknife = (samples[index].sum() - samples[index]) / (n - 1)
            deviations = np.mean(jackknife) - jackknife
            acceleration = float(np.sum(deviations**3) / (6 * np.sum(deviations**2) ** 1.5))
            saved_a = np.asarray(panel["a"])[index, :, 2]
            finite_a = np.isfinite(saved_a)
            check(np.allclose(saved_a[finite_a], acceleration, atol=1e-12, rtol=1e-10), f"outer {index}: acceleration differs from independent jackknife")
            check(np.all(np.isfinite(saved_a[valid[index, :, 2]])), f"outer {index}: valid BCa acceleration is undefined")
    for bi, b in enumerate(b_values):
        for mi, method in enumerate(methods):
            mask = valid[:, bi, mi]
            if method == "t":
                check(not np.any(panel["tail_warning"][:, bi, mi]), "t reference has a bootstrap tail warning")
                continue
            qlo, qhi = np.asarray(panel["q_low"])[:, bi, mi], np.asarray(panel["q_high"])[:, bi, mi]
            check(np.all(np.isfinite(qlo[mask])) and np.all(np.isfinite(qhi[mask]))
                  and np.all((qlo[mask] >= 0) & (qlo[mask] < qhi[mask]) & (qhi[mask] <= 1)), f"{method}/B{b}: invalid quantile range")
            if method == "percentile":
                check(np.allclose(qlo[mask], 0.025) and np.allclose(qhi[mask], 0.975), f"percentile/B{b}: nominal quantiles differ")
            else:
                z0, a = np.asarray(panel["z0"])[:, bi, mi], np.asarray(panel["a"])[:, bi, mi]
                check(np.all(np.isfinite(z0[mask])), f"BCa/B{b}: valid z0 is nonfinite")
                counts = 2 * b * special.ndtr(z0[mask])
                check(np.allclose(counts, np.rint(counts), atol=2e-8, rtol=0), f"BCa/B{b}: z0 is off the tie-count lattice")
                for probability, observed in ((0.025, qlo), (0.975, qhi)):
                    z = stats.norm.ppf(probability)
                    denominator = 1 - a[mask] * (z0[mask] + z)
                    expected_q = special.ndtr(z0[mask] + (z0[mask] + z) / denominator)
                    check(np.all(denominator > 0) and np.allclose(observed[mask], expected_q, atol=2e-12, rtol=1e-10), f"BCa/B{b}: adjusted quantile formula differs")
            low_tau, high_tau = b * np.minimum(qlo, 1 - qlo), b * np.minimum(qhi, 1 - qhi)
            check(np.allclose(np.asarray(panel["tau_low"])[:, bi, mi][mask], low_tau[mask], atol=1e-10, rtol=1e-11), f"{method}/B{b}: lower tau differs")
            check(np.allclose(np.asarray(panel["tau_high"])[:, bi, mi][mask], high_tau[mask], atol=1e-10, rtol=1e-11), f"{method}/B{b}: upper tau differs")
            expected_warning = (low_tau < 10) | (high_tau < 10)
            check(np.array_equal(np.asarray(panel["tail_warning"])[:, bi, mi][mask], expected_warning[mask]), f"{method}/B{b}: warning mask differs")
    return errors


def compare_rows(expected, observed, key_fields, *, label="table"):
    """Compare every independently calculable field, including NA and counts."""
    errors = []

    def key(row):
        return tuple(str(row[field]) for field in key_fields)

    expected_index = {key(row): row for row in expected}
    observed_index = {key(row): row for row in observed}
    if len(expected_index) != len(expected):
        errors.append(f"{label}: duplicate independently reconstructed keys")
    if len(observed_index) != len(observed):
        errors.append(f"{label}: duplicate saved keys")
    if expected_index.keys() != observed_index.keys():
        errors.append(f"{label}: missing/extra keys ({len(expected_index)} expected; {len(observed_index)} saved)")
    for row_key in expected_index.keys() & observed_index.keys():
        actual = observed_index[row_key]
        for field, reference in expected_index[row_key].items():
            if field not in actual:
                errors.append(f"{label}/{row_key}: missing column {field}")
                continue
            value = actual[field]
            if isinstance(reference, (bool, np.bool_)):
                matches = str(value).lower() in (("true", "1") if reference else ("false", "0"))
            elif reference is None or isinstance(reference, (int, float, np.integer, np.floating)):
                target, number = finite_number(reference), finite_number(value)
                if target is None:
                    matches = number is None
                elif number is None:
                    matches = False
                elif isinstance(reference, (int, np.integer)):
                    matches = number == target
                else:
                    matches = np.isclose(number, target, atol=2e-12, rtol=2e-10)
            else:
                matches = str(value) == str(reference)
            if not matches:
                errors.append(f"{label}/{row_key}: {field} differs")
    return errors


def compare_json(expected, observed, *, location="hypotheses"):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [f"{location}: expected object"]
        for key, value in expected.items():
            if key not in observed:
                errors.append(f"{location}: missing {key}")
            else:
                errors.extend(compare_json(value, observed[key], location=f"{location}/{key}"))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(observed, list) or len(expected) != len(observed):
            return [f"{location}: list length/type differs"]
        for index, (first, second) in enumerate(zip(expected, observed)):
            errors.extend(compare_json(first, second, location=f"{location}/{index}"))
    else:
        mismatch = compare_rows([dict(id=0, value=expected)], [dict(id=0, value=observed)], ["id"], label=location)
        if mismatch:
            errors.append(f"{location}: value differs")
    return errors


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def original_metrics(panel):
    samples = np.asarray(panel["samples"], dtype=float)
    se = _original_se(samples)
    output = []
    for oi, outer in enumerate(panel["outer_id"]):
        for bi, b in enumerate(panel["B_values"]):
            for mi, method in enumerate(panel["methods"]):
                count = 1 if method == "t" else len(panel["inner_id"])
                lower, upper = panel["lower"][oi, bi, mi, :count], panel["upper"][oi, bi, mi, :count]
                valid = panel["valid"][oi, bi, mi, :count]
                complete = bool(np.all(valid) and np.isfinite(se[oi]) and se[oi] > 0)
                row = dict(distribution=_meta(panel, "distribution"), n=20, master_seed=_meta(panel, "master_seed"),
                           outer_id=int(outer), B=int(b), method=str(method), R=len(samples), K=len(panel["inner_id"]),
                           alpha=0.05, se=float(se[oi]), sample_sha256=sample_hash(samples[oi]), complete=complete,
                           valid_inner=int(valid.sum()), invalid_inner=int(count - valid.sum()), total_inner=count)
                row.update(dict.fromkeys(METRICS, None))
                if complete:
                    low_sd, high_sd = (0.0, 0.0) if method == "t" else (_sd(lower), _sd(upper))
                    s = float(np.hypot(low_sd, high_sd) / np.sqrt(2))
                    below, above = float(np.mean(upper < 0)), float(np.mean(lower > 0))
                    row.update(lower_sd=low_sd, upper_sd=high_sd, S=s, Ase=s / se[oi], Apop=s * np.sqrt(20),
                               C=float(np.mean((lower <= 0) & (0 <= upper))), Mbelow=below, Mabove=above, T=below - above,
                               mean_width=float(np.mean(upper - lower)), warning_rate=float(np.mean(panel["tail_warning"][oi, bi, mi, :count])))
                output.append(row)
    return output


def metric_summary(values, expected):
    """Keep expected independent units; any missing/invalid input makes main NA."""
    complete = [value for value in values if value is not None and np.isfinite(value)]
    primary = mean_interval(list(values) + [np.nan] * max(0, expected - len(values))) if all(value is not None for value in values) else mean_interval([np.nan] * expected)
    conditional = mean_interval(complete)
    return primary, conditional


def summarize_originals(rows, expected):
    complete = [row for row in rows if row["complete"]]
    output = dict(original_total=expected, observed_original_count=len(rows), missing_original_count=expected - len(rows),
                  complete_original_count=len(complete), incomplete_original_count=expected - len(complete),
                  interval_count=sum(row["total_inner"] for row in rows), valid_count=sum(row["valid_inner"] for row in rows),
                  invalid_count=sum(row["invalid_inner"] for row in rows), conditional_original_count=len(complete))
    for metric in METRICS:
        primary, conditional = metric_summary([row[metric] for row in rows], expected)
        for suffix, key in (("", "mean"), ("_sd", "sd"), ("_mcse", "mcse"), ("_decision_low", "low"), ("_decision_high", "high")):
            output[f"{metric}{suffix}"] = primary[key]
            output[f"conditional_{metric}{suffix}"] = conditional[key]
    return output


def all_summaries(rows):
    seed_rows, pooled_rows = [], []
    for dist in DISTRIBUTIONS:
        for b in BS:
            for method in METHODS:
                for seed in (0, 1, 2, "pooled"):
                    cohort = [row for row in rows if row["distribution"] == dist and row["B"] == b and row["method"] == method
                              and (seed == "pooled" or row["master_seed"] == seed)]
                    expected = 1200 if seed == "pooled" else 400
                    summary = dict(distribution=dist, n=20, master_seed=seed, seed_group="0,1,2" if seed == "pooled" else str(seed),
                                   B=b, method=method, R=expected, K=10, alpha=0.05) | summarize_originals(cohort, expected)
                    (pooled_rows if seed == "pooled" else seed_rows).append(summary)
    return seed_rows, pooled_rows


def quantity_status(interval, threshold, direction, seed_points, *, complete):
    if not complete or not interval["defined"]:
        return "inconclusive"
    low, high = interval["low"], interval["high"]
    if (direction == "less" and low > threshold) or (direction == "greater" and high < threshold):
        return "not_supported"
    if len(seed_points) != 3 or any(value is None or not np.isfinite(value) for value in seed_points):
        return "inconclusive"
    desired = high < threshold if direction == "less" else low > threshold
    seeds_ok = all(point < 0 for point in seed_points) if direction == "less" else all(point > 0 for point in seed_points)
    return "supported" if desired and seeds_ok else "inconclusive"


def confirmation(rows):
    # Only required populations, BCa and B4999 enter either primary hypothesis.
    lookup = {(row["distribution"], row["master_seed"], row["outer_id"]): row for row in rows if row["method"] == "bca" and row["B"] == 4999}

    def values(dist, seed, metric):
        return np.array([lookup[(dist, seed, outer)][metric] if (dist, seed, outer) in lookup and lookup[(dist, seed, outer)]["complete"] else np.nan
                         for outer in range(400)], dtype=float)

    half = [values("gamma_0.5", seed, "C") for seed in (0, 1, 2)]
    sixteen = [values("gamma_16", seed, "C") for seed in (0, 1, 2)]
    directional = [values("gamma_0.5", seed, "T") for seed in (0, 1, 2)]
    h1 = welch_interval(np.concatenate(half), np.concatenate(sixteen))
    h2 = mean_interval(np.concatenate(directional))
    seed_h1 = [float(np.mean(one) - np.mean(zero)) if np.isfinite(one).all() and np.isfinite(zero).all() else None for one, zero in zip(half, sixteen)]
    seed_h2 = [float(np.mean(vector)) if np.isfinite(vector).all() else None for vector in directional]
    h1_complete = all(np.isfinite(vector).all() for vector in half + sixteen)
    h2_complete = all(np.isfinite(vector).all() for vector in directional)
    return {"H1": dict(interval=h1, seed_points=seed_h1, complete=h1_complete,
                       status=quantity_status(h1, -0.02, "less", seed_h1, complete=h1_complete)),
            "H2": dict(interval=h2, seed_points=seed_h2, complete=h2_complete,
                       status=quantity_status(h2, 0.03, "greater", seed_h2, complete=h2_complete))}


def paired_summaries(rows, kind):
    lookup = {(row["distribution"], row["master_seed"], row["outer_id"], row["B"], row["method"]): row for row in rows}
    output = []
    for dist in DISTRIBUTIONS:
        for seed in (0, 1, 2, "pooled"):
            seeds = (0, 1, 2) if seed == "pooled" else (seed,)
            comparisons = [(method, method, 4999, 999) for method in METHODS] if kind == "B" else [
                (method, reference, b, b) for b in BS for method, reference in (("percentile", "t"), ("bca", "percentile"), ("bca", "t"))]
            for method, reference, b, reference_b in comparisons:
                row = dict(distribution=dist, n=20, master_seed=seed, method=method, reference=reference,
                           B=b, reference_B=reference_b, original_total=len(seeds) * 400)
                for metric in ("C", "Ase", "Apop", "T", "mean_width"):
                    values = []
                    for master in seeds:
                        for outer in range(400):
                            current = lookup.get((dist, master, outer, b, method))
                            previous = lookup.get((dist, master, outer, reference_b, reference))
                            values.append(current[metric] - previous[metric] if current and previous and current["complete"] and previous["complete"] else np.nan)
                    primary = mean_interval(values)
                    conditional = mean_interval(np.asarray(values)[np.isfinite(values)])
                    row[f"{metric}_complete_original_count"] = primary["finite_count"]
                    for prefix, result in (("", primary), ("conditional_", conditional)):
                        for suffix, key in (("_difference", "mean"), ("_sd", "sd"), ("_mcse", "mcse"), ("_decision_low", "low"), ("_decision_high", "high")):
                            row[f"{prefix}{metric}{suffix}"] = result[key]
                output.append(row)
    return output


def _mean_reference(interval):
    return {key: interval[key] for key in ("N", "finite_count", "mean", "sd", "mcse", "low", "high", "df")} | dict(valid=interval["defined"])


def _welch_reference(interval):
    return {key: interval[key] for key in ("N1", "N0", "finite_count1", "finite_count0", "mean1", "mean0", "variance1", "variance0", "difference", "df", "low", "high")} | dict(se=interval["mcse"], valid=interval["defined"])


def hypothesis_reference(rows):
    decisions = confirmation(rows)
    lookup = {(row["distribution"], row["master_seed"], row["outer_id"]): row for row in rows if row["method"] == "bca" and row["B"] == 4999}

    def values(dist, seed, metric):
        return np.array([lookup[(dist, seed, outer)][metric] if (dist, seed, outer) in lookup and lookup[(dist, seed, outer)]["complete"] else np.nan for outer in range(400)], dtype=float)

    half = [values("gamma_0.5", seed, "C") for seed in (0, 1, 2)]
    sixteen = [values("gamma_16", seed, "C") for seed in (0, 1, 2)]
    tails = [values("gamma_0.5", seed, "T") for seed in (0, 1, 2)]
    x1, x0, x2 = np.concatenate(half), np.concatenate(sixteen), np.concatenate(tails)
    h1 = _welch_reference(decisions["H1"]["interval"])
    h1.update(status=decisions["H1"]["status"], threshold=-0.02, seed_threshold=0.0, direction="less",
              seed_points=decisions["H1"]["seed_points"], group1="gamma_0.5", group0="gamma_16", independent_groups=True,
              conditional_welch=_welch_reference(welch_interval(x1[np.isfinite(x1)], x0[np.isfinite(x0)])))
    h2 = _mean_reference(decisions["H2"]["interval"])
    h2.update(status=decisions["H2"]["status"], threshold=0.03, seed_threshold=0.0, direction="greater",
              seed_points=decisions["H2"]["seed_points"], conditional_interval=_mean_reference(mean_interval(x2[np.isfinite(x2)])))
    seed_details = [dict(master_seed=seed, H1=_welch_reference(welch_interval(a, b)), H2=_mean_reference(mean_interval(t)))
                    for seed, a, b, t in zip((0, 1, 2), half, sixteen, tails)]
    return dict(question_id="RQ-M04", confirmatory_quantities=2, decision_confidence=0.99,
                independent_originals_per_population=1200, target_method="bca", target_B=4999,
                quantities=dict(H1=h1, H2=h2), seed_details=seed_details,
                hypotheses={name: dict(status=decisions[name]["status"]) for name in ("H1", "H2")})


def audit_evidence(directory):
    directory = Path(directory)
    errors, originals, panel_hashes = [], [], {}
    cells = [(dist, seed) for dist in DISTRIBUTIONS for seed in (0, 1, 2)]
    expected = {f"{dist}_seed{seed}.npz" for dist, seed in cells}
    if {path.name for path in (directory / "panels").glob("*.npz")} != expected:
        errors.append("panels do not contain exactly the 18 preregistered cells")
    for dist, seed in cells:
        name = f"{dist}_seed{seed}.npz"
        path = directory / "panels" / name
        if not path.exists():
            continue
        panel_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with np.load(path, allow_pickle=False) as source:
                panel = {key: source[key] for key in source.files}
            errors.extend(f"{name}: {message}" for message in inspect_panel(panel))
            if (_meta(panel, "distribution"), _meta(panel, "master_seed")) != (dist, seed):
                errors.append(f"{name}: filename and distribution/seed metadata disagree")
            if dist != "normal":
                bound = -np.sqrt(float(dist.split("_")[1]))
                if not np.all(panel["samples"] >= bound):
                    errors.append(f"{name}: standardized-gamma sample is below theoretical support")
            originals.extend(original_metrics(panel))
        except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"{name}: unreadable/inconsistent panel ({type(exc).__name__})")
    summaries, pooled = all_summaries(originals)
    table_counts = {}
    tables = [("original_metrics.csv", originals, ("distribution", "master_seed", "outer_id", "B", "method")),
              ("summary.csv", summaries, ("distribution", "master_seed", "B", "method")),
              ("pooled_summary.csv", pooled, ("distribution", "B", "method")),
              ("paired_B_summary.csv", paired_summaries(originals, "B"), ("distribution", "master_seed", "method", "reference", "B", "reference_B")),
              ("paired_method_summary.csv", paired_summaries(originals, "method"), ("distribution", "master_seed", "method", "reference", "B", "reference_B"))]
    for filename, expected_rows, keys in tables:
        try:
            saved = _csv(directory / filename)
            table_counts[filename] = dict(reconstructed=len(expected_rows), saved=len(saved))
            errors.extend(compare_rows(expected_rows, saved, keys, label=filename))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{filename}: unreadable/inconsistent table ({type(exc).__name__})")
    hypotheses = hypothesis_reference(originals)
    try:
        observed = json.loads((directory / "hypotheses.json").read_text())
        errors.extend(compare_json(hypotheses, observed))
    except (OSError, ValueError) as exc:
        errors.append(f"hypotheses.json: unreadable ({type(exc).__name__})")
    try:
        resources = json.loads((directory / "resources.json").read_text())
        if resources["status"] != "completed" or resources["completed_cells"] != 18 or resources["incomplete_cells"]:
            errors.append("main attempt is not a complete 18-cell panel")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"resources.json: unreadable/inconsistent ({type(exc).__name__})")
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        rng = manifest["randomness"]
        if (rng["main_namespace"] != MAIN_NAMESPACE or rng["distribution_ids"] != dict(zip(DISTRIBUTIONS, range(1, 7)))
                or rng["spawn_key"] != ["namespace", "distribution_id", "n", "outer_id", "purpose_id", "inner_id"]
                or rng["purpose_ids"] != dict(original_sample=0, bootstrap=1) or rng["original_inner_id"] != 0):
            errors.append("manifest RNG namespace/ID/key/purpose metadata differ from preregistration")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"manifest.json: unreadable/inconsistent RNG metadata ({type(exc).__name__})")
    if len(originals) != 43200 or len(summaries) != 108 or len(pooled) != 36:
        errors.append("reconstructed main counts differ from 43200/108/36")
    return dict(question_id="RQ-M04", status="passed" if not errors else "failed", mismatch_count=len(errors), mismatches=errors,
                audit_implementation="independent saved-original/endpoint reconstruction; no primary core or runner imports",
                source_panels_sha256=panel_hashes, original_samples=sum(row["method"] == "t" and row["B"] == 999 for row in originals),
                table_counts=table_counts, hypotheses=hypotheses,
                limitations=["Actual bootstrap prefix reuse, raw rank counts and quantile interpolation cannot be reconstructed without saved bootstrap arrays.",
                             "Common IDs, t/a invariance, q/tau/z0 tie-count lattice, hashes and RNG metadata are checked; actual generation/prefix reuse remains a frozen-source and reference-test responsibility.",
                             "This internal independent implementation audit is not an external scientific review."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/rq_m04"))
    parser.add_argument("--output", type=Path, help="new audit JSON; preserve existing files")
    args = parser.parse_args(argv)
    result = audit_evidence(args.evidence)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"{result['status']}: {result['mismatch_count']} mismatches; {args.output}")
    else:
        print(payload, end="")
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
