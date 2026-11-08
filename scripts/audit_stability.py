"""Independent RQ-M03 audit from saved originals and endpoint diagnostics.

This module deliberately does not import mathscape or the experiment runner.
It never generates observations or bootstrap resamples. Endpoint aggregation,
uncertainty intervals and the preregistered decisions are implemented here
independently. The original bootstrap columns are not retained, so their actual
prefix reuse and the rank counts behind z0 cannot be reconstructed: those are
covered by frozen-source/reference tests, not claimed as empirical audit checks.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import special, stats


METHODS = ("t", "percentile", "bca")
BS = (199, 999, 4999)
DISTRIBUTIONS = ("normal", "centered_exponential")
MAIN_NAMESPACE = 0x4D3033
METRICS = ("lower_sd", "upper_sd", "A", "C", "mean_width", "miss_above", "miss_below", "warning_rate")


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


def mean_interval(values, confidence=0.99):
    """Independent-original mean +/- Student critical value; no clipping."""
    values = np.asarray(values, dtype=float)
    total = values.size
    output = dict(N=total, finite_count=int(np.isfinite(values).sum()), mean=None, sd=None,
                  mcse=None, low=None, high=None, df=total - 1 if total >= 2 else None, defined=False)
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
    output = dict(N1=first.size, N0=second.size, difference=None, mcse=None, df=None,
                  low=None, high=None, defined=False)
    if min(first.size, second.size) < max(2, min_group) or not (np.isfinite(first).all() and np.isfinite(second).all()):
        return output
    difference = float(np.mean(first) - np.mean(second))
    first_var, second_var = _sd(first) ** 2, _sd(second) ** 2
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


def directional_decision(interval, threshold, direction, seed_values, *, complete=True, groups_ok=True):
    """Group adequacy and complete primary inputs precede interval decisions."""
    low, high = interval.get("low"), interval.get("high")
    defined = bool(interval.get("defined")) and low is not None and high is not None
    if not complete or not groups_ok or not defined:
        return "inconclusive"
    if (direction == "less" and low > threshold) or (direction == "greater" and high < threshold):
        return "not_supported"
    if len(seed_values) != 3 or any(value is None or not np.isfinite(value) for value in seed_values):
        return "inconclusive"
    if direction == "less":
        supported = high < threshold and all(value < threshold for value in seed_values)
    else:
        supported = low > threshold and all(value > threshold for value in seed_values)
    return "supported" if supported else "inconclusive"


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
        check(r == 100 and n in (5, 20) and k == 20, "main R/n/K differs from fixed design")
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
    se = np.std(samples, axis=1, ddof=1) / np.sqrt(n)
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


def original_metrics(panel):
    """Recompute original-sample units from endpoints, with no silent deletion."""
    samples = np.asarray(panel["samples"], dtype=float)
    se = np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
    h2 = {row["outer_id"]: row for row in h2_split(panel)} if 999 in panel["B_values"] and len(panel["inner_id"]) == 20 else {}
    rows = []
    for oi, outer_id in enumerate(panel["outer_id"]):
        for bi, b in enumerate(panel["B_values"]):
            for mi, method in enumerate(panel["methods"]):
                k = 1 if method == "t" else np.asarray(panel["inner_id"]).size
                lower = panel["lower"][oi, bi, mi, :k]
                upper = panel["upper"][oi, bi, mi, :k]
                valid = panel["valid"][oi, bi, mi, :k]
                complete = bool(np.all(valid) and np.isfinite(se[oi]) and se[oi] > 0)
                row = dict(distribution=_meta(panel, "distribution"), n=_meta(panel, "n"),
                           master_seed=_meta(panel, "master_seed"), outer_id=int(outer_id),
                           B=int(b), method=str(method), se=float(se[oi]), total_inner=k,
                           valid_inner=int(np.count_nonzero(valid)), invalid_inner=int(k - np.count_nonzero(valid)),
                           complete=complete, sample_sha256=sample_hash(samples[oi]), R=samples.shape[0],
                           K=np.asarray(panel["inner_id"]).size, alpha=_meta(panel, "alpha"))
                row.update(dict.fromkeys(METRICS, None))
                if complete:
                    lower_sd = 0.0 if k == 1 else _sd(lower)
                    upper_sd = 0.0 if k == 1 else _sd(upper)
                    row.update(lower_sd=lower_sd, upper_sd=upper_sd,
                               A=float(np.hypot(lower_sd, upper_sd) / (np.sqrt(2) * se[oi])),
                               C=float(np.mean((lower <= 0) & (0 <= upper))),
                               mean_width=float(np.mean(upper - lower)),
                               miss_above=float(np.mean(lower > 0)), miss_below=float(np.mean(upper < 0)),
                               warning_rate=float(np.mean(panel["tail_warning"][oi, bi, mi, :k])))
                h2_row = h2.get(int(outer_id)) if method == "bca" and b == 999 else None
                row.update(h2_class_complete=bool(h2_row and h2_row["classification_valid"]),
                           h2_eval_complete=bool(h2_row and h2_row["evaluation_valid"]),
                           h2_class_warning_count=h2_row["warning_count"] if h2_row and h2_row["classification_valid"] else None,
                           h2_W=h2_row["W"] if h2_row else None,
                           h2_A_eval=h2_row["A_eval"] if h2_row else None)
                rows.append(row)
    return rows


def _mean_and_mcse(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        return None, None
    return float(np.mean(array)), (float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else None)


def summarize_originals(rows):
    """Aggregate independent-original means; t contributes R intervals only."""
    if not rows:
        raise ValueError("no original metrics")
    complete = [row for row in rows if row["complete"]]
    total_intervals = sum(row["total_inner"] for row in rows)
    valid_intervals = sum(row["valid_inner"] for row in rows)
    output = dict(original_total=len(rows), complete_original_count=len(complete),
                  incomplete_original_count=len(rows) - len(complete), interval_count=total_intervals,
                  valid_count=valid_intervals, invalid_count=total_intervals - valid_intervals)
    for metric in METRICS:
        name = "coverage" if metric == "C" else metric
        reference = mean_interval([row[metric] for row in complete])
        stats_mapping = {"": "mean", "_sd": "sd", "_mcse": "mcse", "_decision_low": "low", "_decision_high": "high"}
        for suffix, key in stats_mapping.items():
            output[f"conditional_{name}{suffix}"] = reference[key]
            output[f"{name}{suffix}"] = reference[key] if len(complete) == len(rows) else None
    output["conditional_original_count"] = len(complete)
    return output


def h2_split(panel):
    """Classification k0:10 and independent evaluation k10:20, never pooled k."""
    bi = list(panel["B_values"]).index(999)
    mi = list(panel["methods"]).index("bca")
    samples = np.asarray(panel["samples"], dtype=float)
    se = np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
    rows = []
    for oi, outer_id in enumerate(panel["outer_id"]):
        valid = panel["valid"][oi, bi, mi]
        classification_valid = valid.size == 20 and bool(np.all(valid[:10]))
        evaluation_valid = valid.size == 20 and bool(np.all(valid[10:20])) and np.isfinite(se[oi]) and se[oi] > 0
        complete = classification_valid and evaluation_valid
        warning_count = int(np.count_nonzero(panel["tail_warning"][oi, bi, mi, :10]))
        evaluation = None
        if evaluation_valid:
            lo = panel["lower"][oi, bi, mi, 10:20]
            hi = panel["upper"][oi, bi, mi, 10:20]
            evaluation = float(np.hypot(_sd(lo), _sd(hi)) / (np.sqrt(2) * se[oi]))
        rows.append(dict(master_seed=_meta(panel, "master_seed"), outer_id=int(outer_id),
                         W=int(warning_count >= 5) if classification_valid else None,
                         warning_count=warning_count, A_eval=evaluation,
                         classification_valid=bool(classification_valid), evaluation_valid=bool(evaluation_valid),
                         complete=bool(complete)))
    return rows


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


def all_summaries(original_rows):
    seed_groups, pooled_groups = {}, {}
    for row in original_rows:
        seed_key = tuple(row[field] for field in ("distribution", "n", "master_seed", "B", "method"))
        pooled_key = tuple(row[field] for field in ("distribution", "n", "B", "method"))
        seed_groups.setdefault(seed_key, []).append(row)
        pooled_groups.setdefault(pooled_key, []).append(row)
    seed_rows = [dict(zip(("distribution", "n", "master_seed", "B", "method"), key)) | summarize_originals(rows)
                 for key, rows in sorted(seed_groups.items())]
    pooled_rows = [dict(zip(("distribution", "n", "B", "method"), key)) | summarize_originals(rows)
                   for key, rows in sorted(pooled_groups.items())]
    for row in seed_rows:
        row.update(seed_group=str(row["master_seed"]), R=100, K=20, alpha=0.05,
                   observed_original_count=row["original_total"], missing_original_count=100 - row["original_total"])
    for row in pooled_rows:
        row.update(master_seed="pooled", seed_group="0,1,2", R=300, K=20, alpha=0.05,
                   observed_original_count=row["original_total"], missing_original_count=300 - row["original_total"])
    return seed_rows, pooled_rows


def confirmation_quantities(original_rows, h2_rows):
    """Four preregistered quantities, independent units and all seed gates."""
    index = {(row["n"], row["master_seed"], row["outer_id"], row["B"]): row
             for row in original_rows if row["distribution"] == "centered_exponential" and row["method"] == "bca"}
    quantities = {}
    for n in (5, 20):
        by_seed = {}
        for seed in (0, 1, 2):
            values = []
            for outer_id in range(100):
                low = index.get((n, seed, outer_id, 199))
                high = index.get((n, seed, outer_id, 4999))
                values.append(high["A"] - 0.5 * low["A"] if low and high and low["complete"] and high["complete"] else np.nan)
            by_seed[seed] = np.asarray(values)
        pooled = mean_interval(np.concatenate(list(by_seed.values())))
        seed_values = [mean_interval(by_seed[seed])["mean"] for seed in (0, 1, 2)]
        quantities[f"H1_n{n}"] = dict(interval=pooled, seed_values=seed_values,
                                      complete=all(np.isfinite(values).all() for values in by_seed.values()),
                                      threshold=0.0, direction="less")
    h3_seeds = {}
    for seed in (0, 1, 2):
        values = []
        for outer_id in range(100):
            row = index.get((5, seed, outer_id, 4999))
            values.append(row["C"] if row and row["complete"] else np.nan)
        h3_seeds[seed] = np.asarray(values)
    quantities["H3"] = dict(interval=mean_interval(np.concatenate(list(h3_seeds.values()))),
                             seed_values=[mean_interval(h3_seeds[seed])["mean"] for seed in (0, 1, 2)],
                             complete=all(np.isfinite(values).all() for values in h3_seeds.values()),
                             threshold=0.93, direction="less")
    h2_index = {(row["master_seed"], row["outer_id"]): row for row in h2_rows}
    complete = all((seed, outer_id) in h2_index and h2_index[(seed, outer_id)]["complete"]
                   for seed in (0, 1, 2) for outer_id in range(100))
    h2_groups, seed_differences = {}, []
    for group in (0, 1, 2, "pooled"):
        members = [row for row in h2_rows if group == "pooled" or row["master_seed"] == group]
        first = [row["A_eval"] for row in members if row["complete"] and row["W"] == 1]
        second = [row["A_eval"] for row in members if row["complete"] and row["W"] == 0]
        interval = welch_interval(first, second, min_group=30 if group == "pooled" else 5)
        h2_groups[group] = interval | dict(total=len(members), complete_count=sum(row["complete"] for row in members))
        if group != "pooled":
            seed_differences.append(interval["difference"])
    groups_ok = all(min(h2_groups[group]["N1"], h2_groups[group]["N0"]) >= (30 if group == "pooled" else 5)
                    for group in (0, 1, 2, "pooled"))
    # Primary Welch interval is undefined if any required original is incomplete.
    h2_primary = h2_groups["pooled"].copy()
    if not complete:
        h2_primary.update(difference=None, mcse=None, df=None, low=None, high=None, defined=False)
    quantities["H2"] = dict(interval=h2_primary, seed_values=seed_differences, complete=complete,
                             groups_ok=groups_ok, groups=h2_groups, threshold=0.02, direction="greater")
    for quantity in quantities.values():
        quantity["status"] = directional_decision(quantity["interval"], quantity["threshold"], quantity["direction"],
                                                   quantity["seed_values"], complete=quantity["complete"],
                                                   groups_ok=quantity.get("groups_ok", True))
    h1_statuses = [quantities[name]["status"] for name in ("H1_n5", "H1_n20")]
    quantities["H1"] = dict(status="not_supported" if "not_supported" in h1_statuses else
                              "supported" if all(value == "supported" for value in h1_statuses) else "inconclusive")
    return quantities


def group_summary(h2_rows):
    output = []
    for seed in (0, 1, 2, "pooled"):
        cohort = [row for row in h2_rows if seed == "pooled" or row["master_seed"] == seed]
        expected = 300 if seed == "pooled" else 100
        complete = len(cohort) == expected and all(row["classification_valid"] and row["evaluation_valid"] for row in cohort)
        for warning_group in (0, 1):
            values = [row["A_eval"] for row in cohort if row["complete"] and row["W"] == warning_group]
            descriptive = mean_interval(values)
            row = dict(master_seed=seed, W=warning_group, original_total=expected,
                       observed_original_count=len(cohort), missing_original_count=expected - len(cohort), group_count=len(values),
                       all_required_complete=complete,
                       unclassified_original_count=sum(not member["classification_valid"] for member in cohort),
                       invalid_evaluation_count=sum(not member["evaluation_valid"] for member in cohort))
            for stat in ("mean", "sd", "mcse"):
                row[f"A_eval_{stat}"] = descriptive[stat] if complete else None
                row[f"conditional_A_eval_{stat}"] = descriptive[stat]
            output.append(row)
    return output


def coverage_difference_rows(original_rows):
    output = []
    lookup = {(row["distribution"], row["n"], row["master_seed"], row["outer_id"], row["B"], row["method"]): row
              for row in original_rows}
    for dist in DISTRIBUTIONS:
        for n in (5, 20):
            for method in METHODS:
                all_values = []
                for seed in (0, 1, 2, "pooled"):
                    if seed == "pooled":
                        values = all_values
                    else:
                        values = []
                        for outer in range(100):
                            low = lookup.get((dist, n, seed, outer, 999, method))
                            high = lookup.get((dist, n, seed, outer, 4999, method))
                            values.append(high["C"] - low["C"] if low and high and low["complete"] and high["complete"] else np.nan)
                        all_values.extend(values)
                    interval = mean_interval(values)
                    output.append(dict(distribution=dist, n=n, method=method, master_seed=seed,
                                       **{key: interval[key] for key in ("N", "finite_count", "mean", "sd", "mcse", "low", "high", "df")},
                                       valid=interval["defined"]))
    return output


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


def hypothesis_reference(quantities):
    output = dict(question_id="RQ-M03", confirmatory_quantities=4, decision_confidence=0.99,
                  pooled_independent_originals=300, quantities={}, hypotheses={})
    for name in ("H1_n5", "H1_n20", "H3"):
        record = quantities[name]
        interval = record["interval"]
        output["quantities"][name] = {key: interval[key] for key in ("N", "finite_count", "mean", "sd", "mcse", "low", "high", "df")}
        output["quantities"][name].update(valid=interval["defined"], seed_points=record["seed_values"],
                                          threshold=record["threshold"], direction=record["direction"], status=record["status"])
    record = quantities["H2"]
    interval = record["interval"]
    output["quantities"]["H2"] = {key: interval[key] for key in ("N1", "N0", "difference", "df", "low", "high")}
    output["quantities"]["H2"].update(se=interval["mcse"], seed_points=record["seed_values"],
                                       threshold=0.02, direction="greater", status=record["status"],
                                       all_required_complete=record["complete"], group_requirements_met=record["groups_ok"],
                                       seed_group_counts_W0_W1=[[record["groups"][seed]["N0"], record["groups"][seed]["N1"]] for seed in (0, 1, 2)])
    output["quantities"]["H2"]["valid"] = bool(interval["defined"] and record["complete"] and record["groups_ok"])
    for name in ("H1", "H2", "H3"):
        output["hypotheses"][name] = dict(status=quantities[name]["status"])
    output["hypotheses"]["H1"].update(n5=quantities["H1_n5"]["status"], n20=quantities["H1_n20"]["status"])
    return output


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_evidence(directory):
    """Audit every predeclared cell and every saved derived numerical table."""
    directory = Path(directory)
    errors, rows, h2_rows, cell_hashes = [], [], [], {}
    expected_cells = [(dist, n, seed) for dist in DISTRIBUTIONS for n in (5, 20) for seed in (0, 1, 2)]
    expected_paths = {f"{dist}_n{n}_seed{seed}.npz" for dist, n, seed in expected_cells}
    found_paths = {path.name for path in (directory / "panels").glob("*.npz")}
    if expected_paths != found_paths:
        errors.append("saved panels do not contain exactly the 12 preregistered cells")
    for dist, n, seed in expected_cells:
        name = f"{dist}_n{n}_seed{seed}.npz"
        path = directory / "panels" / name
        if not path.exists():
            continue
        cell_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with np.load(path, allow_pickle=False) as saved:
                panel = {key: saved[key] for key in saved.files}
            cell_errors = inspect_panel(panel)
            if (_meta(panel, "distribution"), _meta(panel, "n"), _meta(panel, "master_seed")) != (dist, n, seed):
                cell_errors.append("filename and cell metadata disagree")
            errors.extend(f"{name}: {error}" for error in cell_errors)
            rows.extend(original_metrics(panel))
            if dist == "centered_exponential" and n == 20:
                h2_rows.extend(h2_split(panel))
        except (OSError, KeyError, ValueError, IndexError, TypeError) as exc:
            errors.append(f"{name}: unreadable/inconsistent panel ({type(exc).__name__})")
    seed_summary, pooled_summary = all_summaries(rows)
    quantities = confirmation_quantities(rows, h2_rows)
    table_checks = [
        ("original_metrics.csv", rows, ("distribution", "n", "master_seed", "outer_id", "B", "method")),
        ("summary.csv", seed_summary, ("distribution", "n", "master_seed", "B", "method")),
        ("pooled_summary.csv", pooled_summary, ("distribution", "n", "B", "method")),
        ("H2_groups.csv", group_summary(h2_rows), ("master_seed", "W")),
        ("coverage_difference.csv", coverage_difference_rows(rows), ("distribution", "n", "method", "master_seed")),
    ]
    table_counts = {}
    for filename, expected, keys in table_checks:
        try:
            observed = _csv(directory / filename)
            table_counts[filename] = dict(reconstructed=len(expected), saved=len(observed))
            errors.extend(compare_rows(expected, observed, keys, label=filename))
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{filename}: unreadable/inconsistent table ({type(exc).__name__})")
    try:
        hypotheses = json.loads((directory / "hypotheses.json").read_text())
        errors.extend(compare_json(hypothesis_reference(quantities), hypotheses))
    except (OSError, ValueError) as exc:
        errors.append(f"hypotheses.json: unreadable ({type(exc).__name__})")
    try:
        resources = json.loads((directory / "resources.json").read_text())
        if resources["status"] != "completed" or resources["completed_cells"] != 12 or resources["incomplete_cells"]:
            errors.append("main attempt is not a complete 12-cell panel")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"resources.json: unreadable/inconsistent ({type(exc).__name__})")
    expected_original_count = 100 * 12 * 3 * 3
    if len(rows) != expected_original_count or len(seed_summary) != 108 or len(pooled_summary) != 36:
        errors.append("derived table counts differ from preregistered 10800/108/36")
    return dict(question_id="RQ-M03", audit_implementation="independent saved-data reconstruction; no primary aggregator imports",
                status="passed" if not errors else "failed", mismatch_count=len(errors), mismatches=errors,
                source_panels_sha256=cell_hashes, original_samples=sum(row["method"] == "t" and row["B"] == 199 for row in rows),
                table_counts=table_counts, hypotheses=hypothesis_reference(quantities),
                limitations=["No bootstrap arrays are saved: actual prefix reuse, raw z0 rank counts and quantile interpolation values cannot be empirically reconstructed.",
                             "Common IDs/B prefixes, z0 tie-count lattice, BCa q/tau/a formulae and exact t invariance are checked; sequence/prefix generation is covered by frozen-source deterministic tests.",
                             "This is an internal independent implementation audit, not external scientific review."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/rq_m03"))
    parser.add_argument("--output", type=Path, help="new audit JSON; existing files are preserved")
    args = parser.parse_args(argv)
    result = audit_evidence(args.evidence)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"{result['status']}: {result['mismatch_count']} mismatches; {args.output}")
    else:
        print(payload, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
