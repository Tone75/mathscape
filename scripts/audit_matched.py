"""Independent RQ-M05 saved-input audit; no primary model/runner imports.

No random observations or bootstrap samples are generated. Every original-method
record is checked against compact saved endpoints; summaries and independent /
paired contrasts are reconstructed separately. K1 stability quantities are NA.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy import special, stats

DISTRIBUTIONS = ("gamma", "lognormal")
METHODS = ("t", "percentile", "bca")
METRICS = ("C", "Mbelow", "Mabove", "T", "mean_width", "warning_rate")
STABILITY = ("lower_sd", "upper_sd", "S", "Ase", "Apop")
MAIN_NAMESPACE = 0x4D3035
B = 4999
N = 20
R = 10000

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


def finite_number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def compare_row(expected, actual, label):
    errors = []
    for key, target in expected.items():
        if key not in actual:
            errors.append(f"{label}: missing {key}")
            continue
        value = actual[key]
        if isinstance(target, (bool, np.bool_)):
            same = str(value).lower() in (("true", "1") if target else ("false", "0"))
        elif target is None or isinstance(target, (int, float, np.integer, np.floating)):
            first, second = finite_number(target), finite_number(value)
            same = first is None and second is None
            if first is not None and second is not None:
                same = first == second if isinstance(target, (int, np.integer)) else math.isclose(first, second, rel_tol=2e-10, abs_tol=2e-12)
        else:
            same = str(target) == str(value)
        if not same:
            errors.append(f"{label}: {key} differs")
    return errors


def compare_rows(expected, actual, keys, *, label="table"):
    def key(row):
        return tuple(str(row[name]) for name in keys)

    expected_index, actual_index = {key(row): row for row in expected}, {key(row): row for row in actual}
    errors = []
    if len(expected_index) != len(expected) or len(actual_index) != len(actual):
        errors.append(f"{label}: duplicate keys")
    if expected_index.keys() != actual_index.keys():
        errors.append(f"{label}: missing/extra keys")
    for identity in expected_index.keys() & actual_index.keys():
        errors.extend(compare_row(expected_index[identity], actual_index[identity], f"{label}/{identity}"))
    return errors


def _meta(panel, key):
    value = np.asarray(panel[key])
    if value.ndim != 0:
        raise ValueError(f"{key} metadata is not scalar")
    return value.item()


def reasons(panel):
    codes, labels = np.asarray(panel["reason_codes"]), np.asarray(panel["reason_labels"]).astype(str)
    if codes.dtype.kind not in "ui" or (codes < 0).any() or (codes >= labels.size).any():
        raise ValueError("reason codes are outside saved dictionary")
    return labels[codes]


def inspect_panel(panel, *, strict_design=True):
    errors = []
    check = lambda condition, message: errors.append(message) if not condition else None
    samples = np.asarray(panel["samples"], dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 20:
        return ["samples must have shape R,20"]
    total = samples.shape[0]
    check(_meta(panel, "n") == 20 and _meta(panel, "K") == 1 and _meta(panel, "B") == B and _meta(panel, "alpha") == 0.05,
          "n/K/B/alpha metadata differ from fixed design")
    check(_meta(panel, "R") == total, "R metadata differs from saved samples")
    if "observed_original_count" in panel:
        check(_meta(panel, "observed_original_count") == total, "observed count metadata differs")
    check(list(panel["methods"].astype(str)) == list(METHODS), "method order differs")
    check(np.array_equal(panel["outer_id"], np.arange(total)), "original IDs are not complete ordered prefixes")
    if strict_design:
        check(total == R, "main original count differs from 10000")
        check(_meta(panel, "distribution") in DISTRIBUTIONS and _meta(panel, "master_seed") in (0, 1, 2), "population/seed metadata differ")
        check(_meta(panel, "namespace") == MAIN_NAMESPACE, "main namespace differs")
        dist = _meta(panel, "distribution")
        bound = -1.0 if dist == "gamma" else -1 / (2 * np.sinh(np.arcsinh(1.0) / 3))
        check(np.all(samples >= bound), "sample below theoretical population support")
    for field in ("lower", "upper", "z0", "a", "q_low", "q_high", "tau_low", "tau_high", "valid", "tail_warning", "reason_codes"):
        check(np.asarray(panel[field]).shape == (total, 3), f"{field} shape differs from R,3")
    if errors:
        return errors
    reason = reasons(panel)
    valid = panel["valid"]
    lower, upper = panel["lower"], panel["upper"]
    check(valid.dtype.kind == "b" and panel["tail_warning"].dtype.kind == "b", "masks must be boolean")
    check(np.all(np.isfinite(lower[valid])) and np.all(np.isfinite(upper[valid])) and np.all(lower[valid] <= upper[valid]), "valid endpoints are nonfinite/reversed")
    check(np.isnan(lower[~valid]).all() and np.isnan(upper[~valid]).all(), "invalid endpoints contain substitute values")
    check(np.all(reason[valid] == "") and np.all(reason[~valid] != ""), "valid/reason dictionary contract differs")
    hashes = np.asarray(panel["sample_sha256"]).astype(str)
    check(hashes.shape == (total,) and all(hashes[i] == sample_hash(sample) for i, sample in enumerate(samples)), "original hash mismatch")
    se = _original_se(samples)
    original_valid = np.isfinite(se) & (se > 0)
    check(np.array_equal(valid[:, 0], original_valid), "t validity differs from original")
    critical = stats.t.ppf(0.975, 19)
    means = np.mean(samples[original_valid], axis=1)
    check(np.allclose(lower[original_valid, 0], means - critical * se[original_valid], atol=1e-12, rtol=1e-12)
          and np.allclose(upper[original_valid, 0], means + critical * se[original_valid], atol=1e-12, rtol=1e-12), "t reference differs")
    jackknife = (np.sum(samples[original_valid], axis=1, keepdims=True) - samples[original_valid]) / 19
    deviations = np.mean(jackknife, axis=1, keepdims=True) - jackknife
    a = np.sum(deviations**3, axis=1) / (6 * np.sum(deviations**2, axis=1) ** 1.5)
    for mi in range(3):
        saved_a = panel["a"][original_valid, mi]
        finite = np.isfinite(saved_a)
        check(np.allclose(saved_a[finite], a[finite], atol=1e-12, rtol=1e-10), "acceleration differs from independent jackknife")
    check(np.isfinite(panel["a"][valid[:, 2], 2]).all(), "valid BCa acceleration undefined")
    for mi, method in enumerate(METHODS):
        mask = valid[:, mi]
        if method == "t":
            check(not panel["tail_warning"][:, mi].any(), "t has a bootstrap tail warning")
            continue
        qlo, qhi = panel["q_low"][:, mi], panel["q_high"][:, mi]
        check(np.isfinite(qlo[mask]).all() and np.isfinite(qhi[mask]).all()
              and np.all((qlo[mask] >= 0) & (qlo[mask] < qhi[mask]) & (qhi[mask] <= 1)), f"{method}: quantiles invalid")
        if method == "percentile":
            check(np.allclose(qlo[mask], 0.025) and np.allclose(qhi[mask], 0.975), "percentile quantiles differ")
        else:
            z0, acceleration = panel["z0"][mask, mi], panel["a"][mask, mi]
            check(np.isfinite(z0).all(), "BCa z0 nonfinite")
            ranks = 2 * B * special.ndtr(z0)
            check(np.allclose(ranks, np.rint(ranks), atol=2e-8, rtol=0), "BCa z0 off half-tie rank lattice")
            for probability, q in ((0.025, qlo[mask]), (0.975, qhi[mask])):
                z = stats.norm.ppf(probability)
                denominator = 1 - acceleration * (z0 + z)
                expected = special.ndtr(z0 + (z0 + z) / denominator)
                check(np.all(denominator > 0) and np.allclose(q, expected, atol=2e-12, rtol=1e-10), "BCa adjusted quantile formula differs")
        low_tau, high_tau = B * np.minimum(qlo, 1 - qlo), B * np.minimum(qhi, 1 - qhi)
        check(np.allclose(panel["tau_low"][mask, mi], low_tau[mask], atol=1e-10, rtol=1e-11)
              and np.allclose(panel["tau_high"][mask, mi], high_tau[mask], atol=1e-10, rtol=1e-11), f"{method}: tau differs")
        check(np.array_equal(panel["tail_warning"][mask, mi], ((low_tau < 10) | (high_tau < 10))[mask]), f"{method}: warning differs")
    return errors


def original_metrics(panel):
    """Compact arrays; each original/method has exactly one actual interval."""
    se = _original_se(panel["samples"])
    lower, upper = panel["lower"], panel["upper"]
    valid = panel["valid"] & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    complete = valid & np.isfinite(se[:, None]) & (se[:, None] > 0)
    result = dict(se=se, complete=complete, valid_inner=valid.astype(int), invalid_inner=(~valid).astype(int),
                  total_inner=np.ones(valid.shape, dtype=int))
    for metric in METRICS + STABILITY:
        result[metric] = np.full(valid.shape, np.nan)
    for key, value in dict(C=(lower <= 0) & (upper >= 0), Mbelow=upper < 0, Mabove=lower > 0,
                           T=(upper < 0).astype(int) - (lower > 0).astype(int), mean_width=upper - lower,
                           warning_rate=panel["tail_warning"]).items():
        result[key][complete] = value[complete]
    return result


def practical_decision(interval, *, complete=True):
    if not complete or not interval["defined"]:
        return dict(status="inconclusive", direction="undetermined", equivalence_supported=False)
    low, high = interval["low"], interval["high"]
    if low > 0.01:
        return dict(status="supported", direction="positive", equivalence_supported=False)
    if high < -0.01:
        return dict(status="supported", direction="negative", equivalence_supported=False)
    if low > -0.01 and high < 0.01:
        return dict(status="not_supported", direction="within_margin", equivalence_supported=True)
    return dict(status="inconclusive", direction="undetermined", equivalence_supported=False)


def _vector(panels, distribution, seed_group, method, metric):
    """Fixed expected IDs, preserving invalids/missing as NaN and full N."""
    seeds = (0, 1, 2) if seed_group == "pooled" else (seed_group,)
    vectors = []
    for seed in seeds:
        values = np.full(R, np.nan)
        item = panels.get((distribution, seed))
        if item is not None:
            panel, metrics = item
            ids = panel["outer_id"].astype(int)
            mi = METHODS.index(method)
            values[ids] = metrics[metric][:, mi]
        vectors.append(values)
    return np.concatenate(vectors)


def _stats(values):
    return mean_interval(values)


def all_summaries(panels):
    seeded, pooled = [], []
    for dist in DISTRIBUTIONS:
        for seed in (0, 1, 2, "pooled"):
            seeds = (0, 1, 2) if seed == "pooled" else (seed,)
            for method in METHODS:
                mi = METHODS.index(method)
                expected = len(seeds) * R
                cohort = [panels[(dist, master)] for master in seeds if (dist, master) in panels]
                observed = sum(len(panel["outer_id"]) for panel, metrics in cohort)
                complete_count = sum(int(metrics["complete"][:, mi].sum()) for panel, metrics in cohort)
                valid_count = sum(int(metrics["valid_inner"][:, mi].sum()) for panel, metrics in cohort)
                row = dict(distribution=dist, master_seed=seed, original_total=expected, observed_original_count=observed,
                           missing_original_count=expected - observed, complete_original_count=complete_count,
                           interval_count=observed,
                           valid_count=valid_count, invalid_count=observed - valid_count, conditional_original_count=complete_count)
                row.update(n=20, R=expected, K=1, B=4999, alpha=0.05, stability_applicable=False)
                row["incomplete_original_count"] = expected - complete_count
                for metric in METRICS + STABILITY:
                    values = _vector(panels, dist, seed, method, metric)
                    for prefix, result in (("", _stats(values)), ("conditional_", _stats(values[np.isfinite(values)]))):
                        for suffix, key in (("", "mean"), ("_sd", "sd"), ("_mcse", "mcse"), ("_decision_low", "low"), ("_decision_high", "high")):
                            row[f"{prefix}{metric}{suffix}"] = result[key]
                row["method"] = method
                (pooled if seed == "pooled" else seeded).append(row)
    return seeded, pooled


def population_intervals(panels, method, seed):
    return {metric: welch_interval(_vector(panels, "lognormal", seed, method, metric),
                                  _vector(panels, "gamma", seed, method, metric)) for metric in ("C", "T", "mean_width")}


def confirmation(panels):
    result = {}
    for name, metric in (("H1", "C"), ("H2", "T")):
        interval = population_intervals(panels, "bca", "pooled")[metric]
        complete = interval["finite_count1"] == interval["finite_count0"] == 3 * R
        result[name] = dict(interval=interval, **practical_decision(interval, complete=complete))
    return result


def _original_row(panel, metrics, outer_index, mi):
    reason_code = int(panel["reason_codes"][outer_index, mi])
    reason = np.asarray(panel["reason_labels"])[reason_code].item()
    if isinstance(reason, bytes):
        reason = reason.decode("utf-8")
    row = dict(distribution=_meta(panel, "distribution"), master_seed=_meta(panel, "master_seed"),
               outer_id=int(panel["outer_id"][outer_index]), method=METHODS[mi],
               complete=bool(metrics["complete"][outer_index, mi]),
               reason_code=reason_code, reason=str(reason))
    for field in ("valid_inner", "total_inner", "invalid_inner") + METRICS + STABILITY:
        row[field] = metrics[field][outer_index, mi].item()
    return row


def check_original_csv(path, panels):
    errors, seen = [], set()
    # One compact lookup per cell; do not allocate a second 180k-record table.
    indices = {cell: {int(value): i for i, value in enumerate(panel["outer_id"])} for cell, (panel, metrics) in panels.items()}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identity = (row["distribution"], int(row["master_seed"]), int(row["outer_id"]), row["method"])
            if identity in seen:
                errors.append(f"original_metrics.csv.gz/{identity}: duplicate row")
            seen.add(identity)
            cell = identity[:2]
            if cell not in panels or identity[2] not in indices[cell] or identity[3] not in METHODS:
                errors.append(f"original_metrics.csv.gz/{identity}: unexpected identity")
                continue
            panel, metrics = panels[cell]
            reference = _original_row(panel, metrics, indices[cell][identity[2]], METHODS.index(identity[3]))
            errors.extend(compare_row(reference, row, f"original_metrics.csv.gz/{identity}"))
    expected = sum(len(panel["outer_id"]) * 3 for panel, metrics in panels.values())
    if len(seen) != expected or len(seen) != 180000:
        errors.append("original_metrics.csv.gz: count differs from reconstructed/predeclared 180000")
    return errors, len(seen)


def _mean_reference(interval):
    return {key: interval[key] for key in ("N", "finite_count", "mean", "sd", "mcse", "low", "high", "df")} | dict(valid=interval["defined"])


def _welch_reference(interval):
    return {key: interval[key] for key in ("N1", "N0", "finite_count1", "finite_count0", "mean1", "mean0", "variance1", "variance0", "difference", "df", "low", "high")} | dict(se=interval["mcse"], valid=interval["defined"])


def population_contrasts(panels):
    rows = []
    for seed in (0, 1, 2, "pooled"):
        for method in METHODS:
            row = dict(master_seed=seed, method=method, n=20, K=1, B=4999, alpha=0.05,
                       group1="lognormal", group0="gamma", independent_groups=True)
            for metric in ("C", "T", "mean_width"):
                one, zero = _vector(panels, "lognormal", seed, method, metric), _vector(panels, "gamma", seed, method, metric)
                for prefix, first, second in (("", one, zero), ("conditional_", one[np.isfinite(one)], zero[np.isfinite(zero)])):
                    for key, value in _welch_reference(welch_interval(first, second)).items():
                        row[f"{prefix}{metric}_{key}"] = value
            rows.append(row)
    return rows


def method_contrasts(panels):
    rows = []
    for dist in DISTRIBUTIONS:
        for seed in (0, 1, 2, "pooled"):
            for method, reference in (("percentile", "t"), ("bca", "percentile"), ("bca", "t")):
                row = dict(distribution=dist, master_seed=seed, method=method, reference=reference,
                           n=20, K=1, B=4999, alpha=0.05, original_total=3 * R if seed == "pooled" else R)
                for metric in ("C", "T", "mean_width"):
                    difference = _vector(panels, dist, seed, method, metric) - _vector(panels, dist, seed, reference, metric)
                    for prefix, values in (("", difference), ("conditional_", difference[np.isfinite(difference)])):
                        for key, value in _mean_reference(mean_interval(values)).items():
                            row[f"{prefix}{metric}_{key}"] = value
                rows.append(row)
    return rows


def hypothesis_reference(panels):
    quantities = {}
    for name, metric in (("H1", "C"), ("H2", "T")):
        one, zero = _vector(panels, "lognormal", "pooled", "bca", metric), _vector(panels, "gamma", "pooled", "bca", metric)
        interval = welch_interval(one, zero)
        quantities[name] = _welch_reference(interval) | practical_decision(interval)
        quantities[name].update(metric=metric, margin=0.01, group1="lognormal", group0="gamma", independent_groups=True,
                                conditional_welch=_welch_reference(welch_interval(one[np.isfinite(one)], zero[np.isfinite(zero)])))
    seed_details = [dict(master_seed=seed, **{name: _welch_reference(population_intervals(panels, "bca", seed)[metric])
                                             for name, metric in (("H1", "C"), ("H2", "T"))}) for seed in (0, 1, 2)]
    return dict(question_id="RQ-M05", confirmatory_quantities=2, decision_confidence=0.99, margin=0.01,
                target_method="bca", B=4999, independent_originals_per_population=30000, seed_direction_gate=False,
                quantities=quantities, seed_details=seed_details,
                hypotheses={name: {key: quantity[key] for key in ("status", "direction", "equivalence_supported")} for name, quantity in quantities.items()},
                both_metrics_equivalent=all(quantity["equivalence_supported"] for quantity in quantities.values()))


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_evidence(directory):
    directory = Path(directory)
    errors, panels, panel_hashes = [], {}, {}
    cells = [(dist, seed) for dist in DISTRIBUTIONS for seed in (0, 1, 2)]
    expected_files = {f"{dist}_seed{seed}.npz" for dist, seed in cells}
    if {path.name for path in (directory / "panels").glob("*.npz")} != expected_files:
        errors.append("saved panels differ from the six predeclared cells")
    for dist, seed in cells:
        path = directory / "panels" / f"{dist}_seed{seed}.npz"
        if not path.exists():
            continue
        panel_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with np.load(path, allow_pickle=False) as saved:
                panel = {key: saved[key] for key in saved.files}
            errors.extend(f"{path.name}: {message}" for message in inspect_panel(panel))
            if (_meta(panel, "distribution"), _meta(panel, "master_seed")) != (dist, seed):
                errors.append(f"{path.name}: filename and cell metadata disagree")
            panels[(dist, seed)] = (panel, original_metrics(panel))
        except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
            errors.append(f"{path.name}: unreadable/inconsistent ({type(exc).__name__})")
    table_counts = {}
    try:
        row_errors, total = check_original_csv(directory / "original_metrics.csv.gz", panels)
        errors.extend(row_errors)
        table_counts["original_metrics.csv.gz"] = dict(reconstructed=sum(len(panel["outer_id"]) * 3 for panel, values in panels.values()), saved=total)
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        errors.append(f"original_metrics.csv.gz: unreadable/inconsistent ({type(exc).__name__})")
    seed_rows, pooled = all_summaries(panels)
    for filename, expected_rows, keys in (
        ("summary.csv", seed_rows, ("distribution", "master_seed", "method")),
        ("pooled_summary.csv", pooled, ("distribution", "method")),
        ("population_contrasts.csv", population_contrasts(panels), ("master_seed", "method")),
        ("paired_method_contrasts.csv", method_contrasts(panels), ("distribution", "master_seed", "method", "reference")),
    ):
        try:
            saved = _csv(directory / filename)
            table_counts[filename] = dict(reconstructed=len(expected_rows), saved=len(saved))
            errors.extend(compare_rows(expected_rows, saved, keys, label=filename))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            errors.append(f"{filename}: unreadable/inconsistent ({type(exc).__name__})")
    hypotheses = hypothesis_reference(panels)
    try:
        errors.extend(compare_json(hypotheses, json.loads((directory / "hypotheses.json").read_text())))
    except (OSError, ValueError) as exc:
        errors.append(f"hypotheses.json: unreadable ({type(exc).__name__})")
    try:
        resources = json.loads((directory / "resources.json").read_text())
        if resources["status"] != "completed" or resources["completed_cells"] != 6 or resources["incomplete_cells"]:
            errors.append("main attempt is not a completed six-cell experiment")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"resources.json: unreadable/inconsistent ({type(exc).__name__})")
    try:
        rng = json.loads((directory / "manifest.json").read_text())["randomness"]
        if (rng["main_namespace"] != MAIN_NAMESPACE or rng["distribution_ids"] != {"gamma": 1, "lognormal": 2}
                or rng["spawn_key"] != ["namespace", "distribution_id", "n", "outer_id", "purpose_id", "inner_id"]
                or rng["inner_id"] != 0 or rng["purpose_ids"] != dict(original_sample=0, bootstrap=1)
                or rng["entropy"] != "master_seed" or rng["smoke_namespace"] != 0x534D3035):
            errors.append("RNG namespace/IDs/spawn metadata differ from preregistration")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"manifest.json: unreadable/inconsistent RNG metadata ({type(exc).__name__})")
    original_count = sum(len(panel["outer_id"]) for panel, metrics in panels.values())
    if original_count != 60000:
        errors.append("saved original count differs from 60000")
    return dict(question_id="RQ-M05", status="passed" if not errors else "failed", mismatch_count=len(errors), mismatches=errors,
                audit_implementation="independent saved-input reconstruction; no primary core or runner imports; gzip records checked one at a time",
                original_samples=original_count, source_panels_sha256=panel_hashes, table_counts=table_counts, hypotheses=hypotheses,
                limitations=["Bootstrap draws, rank counts and interpolation cannot be reconstructed without saved bootstrap arrays; shared percentile/BCa resampling is a frozen-source/reference-test responsibility.",
                             "All saved originals/endpoint diagnostics, hashes, t/a/q/tau/rank-lattice consistency, 180000 records, summaries, contrasts and decisions are independently checked.",
                             "K1 stability SD/S/Ase/Apop are not estimated, including t. This is an internal audit, not external scientific review."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/rq_m05"))
    parser.add_argument("--output", type=Path, help="new audit JSON; existing file is preserved")
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
