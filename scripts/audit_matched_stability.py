"""Independent RQ-M06 saved-original/endpoint and designated-stream audit.

No primary model/runner imports and no random-number generation. Every saved
endpoint is aggregated independently. Exactly the 18 preregistered streams also
permit index/mean/prefix/quantile reconstruction; other bootstrap generation
processes are outside the saved-array reconstruction scope.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy import special, stats

DISTRIBUTIONS = ("gamma", "lognormal")
METHODS = ("t", "percentile", "bca")
BS = (999, 4999)
R, K, N = 200, 30, 20
MAIN_NAMESPACE = 0x4D3036
ACCEPTED_INPUT = "dc2501c3099f5c25227d9eee00d227c2f0b02f81"
METRICS = ("lower_sd", "upper_sd", "S", "Ase", "Apop", "Vse", "C", "Mbelow", "Mabove", "T", "mean_width", "warning_rate")
ROOT = Path(__file__).resolve().parents[1]

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


def _reason_array(panel):
    codes, labels = panel["reason_codes"], panel["reason_labels"].astype(str)
    if codes.dtype.kind not in "ui" or np.any(codes < 0) or np.any(codes >= labels.size):
        raise ValueError("reason code is outside the saved dictionary")
    return labels[codes]


def reference_intervals(sample, means):
    """Independent fixed-array reference; no statistical model imports or RNG."""
    sample, means = np.asarray(sample, dtype=float), np.asarray(means, dtype=float)
    empty = dict(lower=np.nan, upper=np.nan, z0=np.nan, a=np.nan, q_low=np.nan, q_high=np.nan,
                 tau_low=np.nan, tau_high=np.nan, valid=False, tail_warning=False)
    results = {method: empty.copy() for method in METHODS}
    if sample.shape != (20,) or not np.isfinite(sample).all() or np.all(sample == sample[0]):
        return results
    mean, se = float(np.mean(sample)), float(_original_se(sample[None, :])[0])
    if not np.isfinite(se) or se <= 0:
        return results
    jackknife = (sample.sum() - sample) / 19
    centered = jackknife.mean() - jackknife
    a = float(np.sum(centered**3) / (6 * np.sum(centered**2) ** 1.5))
    critical = stats.t.ppf(0.975, 19)
    results["t"].update(lower=mean - critical * se, upper=mean + critical * se, valid=True, a=a)
    results["percentile"]["a"] = results["bca"]["a"] = a
    if means.ndim != 1 or means.size == 0 or not np.isfinite(means).all():
        return results

    def record(qlo, qhi, **diagnostics):
        if not np.isfinite([qlo, qhi]).all() or not 0 <= qlo < qhi <= 1:
            return empty | diagnostics
        lower, upper = np.quantile(means, [qlo, qhi], method="linear")
        low_tau, high_tau = means.size * min(qlo, 1 - qlo), means.size * min(qhi, 1 - qhi)
        return empty | dict(lower=float(lower), upper=float(upper), valid=True, q_low=float(qlo), q_high=float(qhi),
                            tau_low=low_tau, tau_high=high_tau, tail_warning=min(low_tau, high_tau) < 10) | diagnostics

    results["percentile"] = record(0.025, 0.975, a=a)
    probability = (np.count_nonzero(means < mean) + 0.5 * np.count_nonzero(means == mean)) / means.size
    z0 = float(special.ndtri(probability))
    if not np.isfinite([a, z0]).all():
        results["bca"].update(a=a, z0=z0)
        return results
    z = stats.norm.ppf([0.025, 0.975])
    denominator = 1 - a * (z0 + z)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        results["bca"].update(a=a, z0=z0)
        return results
    q = special.ndtr(z0 + (z0 + z) / denominator)
    results["bca"] = record(q[0], q[1], a=a, z0=z0)
    return results


def inspect_panel(panel, *, strict_design=True):
    errors = []
    check = lambda condition, message: errors.append(message) if not condition else None
    samples = np.asarray(panel["samples"], dtype=float)
    total, n = samples.shape
    b_values, methods = list(panel["B_values"]), list(panel["methods"].astype(str))
    inner = panel["inner_id"]
    k = inner.size
    shape = (total, len(b_values), len(methods), k)
    check(n == 20 and _meta(panel, "n") == 20 and _meta(panel, "alpha") == 0.05, "n/alpha fixed metadata differ")
    check(np.array_equal(panel["outer_id"], np.arange(total)), "outer IDs are not first ordered IDs")
    check(np.array_equal(inner, np.arange(k)), "inner IDs are not ordered common IDs")
    check(methods == list(METHODS) and b_values == list(BS), "method/B axis differs from fixed design")
    if strict_design:
        check(total == R and k == K and _meta(panel, "R") == R and _meta(panel, "K") == K, "main R200/K30 metadata differ")
        check(_meta(panel, "namespace") == MAIN_NAMESPACE and _meta(panel, "distribution") in DISTRIBUTIONS
              and _meta(panel, "master_seed") in (0, 1, 2), "main namespace/population/seed differ")
    for field in ("lower", "upper", "a", "z0", "q_low", "q_high", "tau_low", "tau_high", "valid", "tail_warning", "reason_codes"):
        check(panel[field].shape == shape, f"{field} shape differs from common R/B/method/inner grid")
    if errors:
        return errors
    valid, lower, upper = panel["valid"], panel["lower"], panel["upper"]
    reason = _reason_array(panel)
    check(valid.dtype.kind == "b" and panel["tail_warning"].dtype.kind == "b", "masks are not boolean")
    check(np.isfinite(lower[valid]).all() and np.isfinite(upper[valid]).all() and np.all(lower[valid] <= upper[valid]), "valid endpoints nonfinite/reversed")
    check(np.isnan(lower[~valid]).all() and np.isnan(upper[~valid]).all(), "invalid endpoints were substituted")
    check(np.all(reason[valid] == "") and np.all(reason[~valid] != ""), "valid/reason dictionary mismatch")
    hashes = panel["sample_sha256"].astype(str)
    check(hashes.shape == (total,) and all(hashes[i] == sample_hash(sample) for i, sample in enumerate(samples)), "original sample hash mismatch")
    se = _original_se(samples)
    original_valid = np.isfinite(se) & (se > 0)
    critical = stats.t.ppf(0.975, 19)
    for i in range(total):
        check(np.all(valid[i, :, 0] == original_valid[i]), f"outer{i}: t validity differs")
        if not original_valid[i]:
            continue
        mean = np.mean(samples[i])
        check(np.allclose(lower[i, :, 0], mean - critical * se[i], atol=1e-12, rtol=1e-12)
              and np.allclose(upper[i, :, 0], mean + critical * se[i], atol=1e-12, rtol=1e-12), f"outer{i}: t reference differs")
        check(np.all(lower[i, :, 0] == lower[i, 0, 0, 0]) and np.all(upper[i, :, 0] == upper[i, 0, 0, 0]), f"outer{i}: t changes across B/inner")
        jackknife = (samples[i].sum() - samples[i]) / 19
        deviations = jackknife.mean() - jackknife
        a = np.sum(deviations**3) / (6 * np.sum(deviations**2) ** 1.5)
        saved = panel["a"][i]
        finite = np.isfinite(saved)
        check(np.allclose(saved[finite], a, atol=1e-12, rtol=1e-10), f"outer{i}: acceleration differs from independent jackknife")
        check(np.isfinite(panel["a"][i, :, 2][valid[i, :, 2]]).all(), f"outer{i}: valid BCa acceleration undefined")
    for bi, b in enumerate(b_values):
        for mi, method in enumerate(METHODS):
            mask = valid[:, bi, mi]
            if method == "t":
                check(not panel["tail_warning"][:, bi, mi].any(), "t has bootstrap tail warning")
                continue
            qlo, qhi = panel["q_low"][:, bi, mi], panel["q_high"][:, bi, mi]
            check(np.isfinite(qlo[mask]).all() and np.isfinite(qhi[mask]).all()
                  and np.all((qlo[mask] >= 0) & (qlo[mask] < qhi[mask]) & (qhi[mask] <= 1)), f"{method}/B{b}: invalid quantiles")
            if method == "percentile":
                check(np.allclose(qlo[mask], 0.025) and np.allclose(qhi[mask], 0.975), f"percentile/B{b}: quantiles differ")
            else:
                z0, a = panel["z0"][:, bi, mi][mask], panel["a"][:, bi, mi][mask]
                check(np.isfinite(z0).all(), f"BCa/B{b}: nonfinite valid z0")
                ranks = 2 * b * special.ndtr(z0)
                check(np.allclose(ranks, np.rint(ranks), atol=2e-8, rtol=0), f"BCa/B{b}: z0 off half-tie rank grid")
                for probability, observed in ((0.025, qlo[mask]), (0.975, qhi[mask])):
                    z = stats.norm.ppf(probability)
                    den = 1 - a * (z0 + z)
                    expected = special.ndtr(z0 + (z0 + z) / den)
                    check(np.all(den > 0) and np.allclose(observed, expected, atol=2e-12, rtol=1e-10), f"BCa/B{b}: adjusted quantile formula differs")
            lo_tau, hi_tau = b * np.minimum(qlo, 1 - qlo), b * np.minimum(qhi, 1 - qhi)
            check(np.allclose(panel["tau_low"][:, bi, mi][mask], lo_tau[mask], atol=1e-10, rtol=1e-11)
                  and np.allclose(panel["tau_high"][:, bi, mi][mask], hi_tau[mask], atol=1e-10, rtol=1e-11), f"{method}/B{b}: tau differs")
            check(np.array_equal(panel["tail_warning"][:, bi, mi][mask], ((lo_tau < 10) | (hi_tau < 10))[mask]), f"{method}/B{b}: warning differs")
    return errors


def original_metrics(panel):
    se = _original_se(panel["samples"])
    rows = []
    for oi, outer in enumerate(panel["outer_id"]):
        for bi, b in enumerate(panel["B_values"]):
            for mi, method in enumerate(panel["methods"].astype(str)):
                k = 1 if method == "t" else len(panel["inner_id"])
                lower, upper, valid = (panel[field][oi, bi, mi, :k] for field in ("lower", "upper", "valid"))
                complete = bool(valid.all() and np.isfinite(se[oi]) and se[oi] > 0)
                row = dict(distribution=_meta(panel, "distribution"), master_seed=_meta(panel, "master_seed"), outer_id=int(outer),
                           B=int(b), method=str(method), se=float(se[oi]), sample_sha256=sample_hash(panel["samples"][oi]),
                           n=20, R=len(panel["samples"]), K=len(panel["inner_id"]), alpha=0.05,
                           complete=complete, valid_inner=int(valid.sum()), total_inner=k, invalid_inner=int(k - valid.sum()))
                row.update(dict.fromkeys(METRICS, None))
                if complete:
                    lsd, usd = (0.0, 0.0) if method == "t" else (_sd(lower), _sd(upper))
                    s = float(np.hypot(lsd, usd) / np.sqrt(2))
                    below, above = float(np.mean(upper < 0)), float(np.mean(lower > 0))
                    row.update(lower_sd=lsd, upper_sd=usd, S=s, Ase=s / se[oi], Apop=s * np.sqrt(20), Vse=(s / se[oi]) ** 2,
                               C=float(np.mean((lower <= 0) & (upper >= 0))), Mbelow=below, Mabove=above, T=below - above,
                               mean_width=float(np.mean(upper - lower)), warning_rate=float(np.mean(panel["tail_warning"][oi, bi, mi, :k])))
                rows.append(row)
    return rows


def audit_inputs(inputs, panels):
    errors = []
    required = {"evidence/rq_m05/manifest.json"} | {f"evidence/rq_m05/panels/{dist}_seed{seed}.npz" for dist in DISTRIBUTIONS for seed in (0, 1, 2)}
    if (inputs.get("accepted_head") != ACCEPTED_INPUT or inputs.get("question_id") != "RQ-M06"
            or inputs.get("source_question_id") != "RQ-M05" or inputs.get("n") != 20
            or inputs.get("originals_per_cell") != 200 or inputs.get("selected_originals") != 1200):
        errors.append("input provenance fixed metadata differ")
    if set(inputs.get("source_files_sha256", {})) != required:
        errors.append("input provenance file set differs from accepted M05 manifest plus six panels")
    selections = {(row["distribution"], row["master_seed"]): row for row in inputs.get("selections", [])}
    if len(selections) != 6 or len(inputs.get("selections", [])) != 6:
        errors.append("input provenance selection cells missing/duplicated")
    for relative in sorted(required):
        try:
            payload = (ROOT / relative).read_bytes()
            if hashlib.sha256(payload).hexdigest() != inputs.get("source_files_sha256", {}).get(relative):
                errors.append(f"{relative}: accepted input SHA256 differs")
            accepted_blob = subprocess.run(["git", "rev-parse", f"{ACCEPTED_INPUT}:{relative}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
            actual_blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            if accepted_blob != actual_blob:
                errors.append(f"{relative}: bytes differ from accepted Git blob")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{relative}: cannot verify accepted input ({type(exc).__name__})")
    for dist in DISTRIBUTIONS:
        for seed in (0, 1, 2):
            identity = (dist, seed)
            row = selections.get(identity)
            path = f"evidence/rq_m05/panels/{dist}_seed{seed}.npz"
            if row is None:
                errors.append(f"{identity}: input selection missing")
                continue
            try:
                with np.load(ROOT / path, allow_pickle=False) as source:
                    samples, hashes = source["samples"], source["sample_sha256"].astype(str)
                    if samples.shape != (10000, 20) or not np.array_equal(source["outer_id"], np.arange(10000)):
                        errors.append(f"{identity}: source M05 IDs/shape differ")
                    if any(hashes[i] != sample_hash(sample) for i, sample in enumerate(samples)):
                        errors.append(f"{identity}: source M05 original hash differs")
                    chosen = samples[:200].copy()
                    chosen_hashes = hashes[:200].tolist()
                    if (row["source_path"] != path or row["outer_ids"] != list(range(200))
                            or row["sample_sha256"] != chosen_hashes
                            or row["selected_samples_sha256"] != sample_hash(chosen)):
                        errors.append(f"{identity}: preselected M05 IDs/hash map differs")
                if identity in panels:
                    saved = panels[identity]
                    if not np.array_equal(saved["samples"], chosen) or not np.array_equal(saved["outer_id"], np.arange(200)):
                        errors.append(f"{identity}: M06 originals differ from accepted M05 selection")
            except (OSError, KeyError, ValueError, IndexError) as exc:
                errors.append(f"{identity}: cannot verify selected source ({type(exc).__name__})")
    return errors


def audit_stream(path, panel, dist, seed, inner):
    errors = []
    check = lambda condition, message: errors.append(message) if not condition else None
    with np.load(path, allow_pickle=False) as source:
        stream = {key: source[key] for key in source.files}
    indices, means = stream["indices"], stream["means"]
    check(indices.shape == (4999, 20) and indices.dtype == np.dtype("uint8"), "indices shape/dtype differ")
    check(means.shape == (4999,) and means.dtype == np.dtype("float64"), "means shape/dtype differ")
    if errors:
        return errors
    check(np.all(indices < 20), "resampling index outside original sample")
    if np.any(indices >= 20):
        return errors
    for key, value in dict(distribution=dist, master_seed=seed, outer_id=0, inner_id=inner, n=20, B=4999, K=30,
                           alpha=0.05, namespace=MAIN_NAMESPACE, native_indices_dtype="int64", stored_indices_dtype="uint8").items():
        check(_meta(stream, key) == value, f"stream {key} metadata differs")
    check(np.array_equal(stream["B_values"], BS), "stream prefix metadata differ")
    digest = _meta(stream, "sample_sha256")
    digest = digest.decode("ascii") if isinstance(digest, bytes) else str(digest)
    sample = panel["samples"][0]
    check(digest == sample_hash(sample), "stream original hash differs")
    actual_means = np.mean(sample[indices], axis=1)
    check(np.array_equal(actual_means, means), "saved bootstrap means differ from saved indices/original")
    for bi, b in enumerate(BS):
        reference = reference_intervals(sample, actual_means[:b])
        for mi, method in enumerate(METHODS):
            saved = {field: panel[field][0, bi, mi, inner].item() for field in reference[method]}
            errors.extend(compare_rows([dict(id=0, **reference[method])], [dict(id=0, **saved)], ["id"], label=f"B{b}/{method}"))
    return errors


def _metric_stats(values):
    values = np.asarray(values, dtype=float)
    interval = mean_interval(values)
    output = {"": interval["mean"], "_sd": interval["sd"], "_mcse": interval["mcse"],
              "_decision_low": interval["low"], "_decision_high": interval["high"]}
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear") if values.size and np.isfinite(values).all() else [None] * 5
    output.update(dict(zip(("_p05", "_p25", "_p50", "_p75", "_p95"), quantiles)))
    return output


def summarize_originals(rows, expected):
    complete = [row for row in rows if row["complete"]]
    output = dict(original_total=expected, observed_original_count=len(rows), missing_original_count=expected - len(rows),
                  complete_original_count=len(complete), incomplete_original_count=expected - len(complete),
                  interval_count=sum(row["total_inner"] for row in rows), valid_count=sum(row["valid_inner"] for row in rows),
                  invalid_count=sum(row["invalid_inner"] for row in rows), conditional_original_count=len(complete))
    for metric in METRICS:
        primary = [row[metric] if row["complete"] else np.nan for row in rows] + [np.nan] * max(0, expected - len(rows))
        conditional = [row[metric] for row in complete]
        for prefix, values in (("", primary), ("conditional_", conditional)):
            for suffix, value in _metric_stats(values).items():
                output[f"{prefix}{metric}{suffix}"] = value
    return output


def all_summaries(rows):
    seed_rows, pooled = [], []
    for dist in DISTRIBUTIONS:
        for seed in (0, 1, 2, "pooled"):
            expected = R * 3 if seed == "pooled" else R
            for b in BS:
                for method in METHODS:
                    cohort = [row for row in rows if row["distribution"] == dist and row["B"] == b and row["method"] == method
                              and (seed == "pooled" or row["master_seed"] == seed)]
                    row = dict(distribution=dist, master_seed=seed, n=20, R=expected, K=30, alpha=0.05, B=b, method=method) | summarize_originals(cohort, expected)
                    (pooled if seed == "pooled" else seed_rows).append(row)
    return seed_rows, pooled


def _expected(rows, dist, seed, b, method, metric):
    index = {row["outer_id"]: row for row in rows if row["distribution"] == dist and row["master_seed"] == seed and row["B"] == b and row["method"] == method}
    return np.array([index[outer][metric] if outer in index and index[outer]["complete"] else np.nan for outer in range(R)], dtype=float)


def _vector(rows, dist, seed, b, method, metric):
    seeds = (0, 1, 2) if seed == "pooled" else (seed,)
    return np.concatenate([_expected(rows, dist, master, b, method, metric) for master in seeds])


def practical_decision(interval, complete=True):
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


def reduction_decision(interval, complete=True):
    if not complete or not interval["defined"]:
        return "inconclusive"
    if interval["high"] < 0:
        return "supported"
    if interval["low"] >= 0:
        return "not_supported"
    return "inconclusive"


def combine_reduction(statuses):
    if "not_supported" in statuses:
        return "not_supported"
    return "supported" if list(statuses) == ["supported", "supported"] else "inconclusive"


def confirmation(rows):
    one = _vector(rows, "lognormal", "pooled", 4999, "bca", "Ase")
    zero = _vector(rows, "gamma", "pooled", 4999, "bca", "Ase")
    interval = welch_interval(one, zero)
    output = dict(H1=dict(interval=interval, **practical_decision(interval)))
    for dist in DISTRIBUTIONS:
        low = _vector(rows, dist, "pooled", 999, "bca", "Ase")
        high = _vector(rows, dist, "pooled", 4999, "bca", "Ase")
        reference = mean_interval(high - 0.75 * low)
        ratio = float(high.mean() / low.mean()) if np.isfinite(high).all() and np.isfinite(low).all() and low.mean() != 0 else None
        output[f"H2_{dist}"] = dict(interval=reference, status=reduction_decision(reference), ratio_of_means=ratio)
    output["H2"] = dict(status=combine_reduction([output[f"H2_{dist}"]["status"] for dist in DISTRIBUTIONS]))
    return output


def _mean_reference(interval):
    return {key: interval[key] for key in ("N", "finite_count", "mean", "sd", "mcse", "low", "high", "df")} | dict(valid=interval["defined"])


def _welch_reference(interval):
    return {key: interval[key] for key in ("N1", "N0", "finite_count1", "finite_count0", "mean1", "mean0", "variance1", "variance0", "difference", "df", "low", "high")} | dict(se=interval["mcse"], valid=interval["defined"])


def population_contrasts(rows):
    output = []
    for seed in (0, 1, 2, "pooled"):
        for b in BS:
            for method in METHODS:
                row = dict(master_seed=seed, B=b, method=method, n=20, K=30, alpha=0.05, group1="lognormal", group0="gamma", independent_groups=True)
                for metric in METRICS:
                    one, zero = _vector(rows, "lognormal", seed, b, method, metric), _vector(rows, "gamma", seed, b, method, metric)
                    for prefix, first, second in (("", one, zero), ("conditional_", one[np.isfinite(one)], zero[np.isfinite(zero)])):
                        for key, value in _welch_reference(welch_interval(first, second)).items():
                            row[f"{prefix}{metric}_{key}"] = value
                output.append(row)
    return output


def paired_contrasts(rows, kind):
    output = []
    for dist in DISTRIBUTIONS:
        for seed in (0, 1, 2, "pooled"):
            comparisons = [(m, m, 4999, 999) for m in METHODS] if kind == "B" else [
                (m, ref, b, b) for b in BS for m, ref in (("percentile", "t"), ("bca", "percentile"), ("bca", "t"))]
            for method, reference, b, reference_b in comparisons:
                row = dict(distribution=dist, master_seed=seed, method=method, reference=reference, B=b, reference_B=reference_b,
                           n=20, K=30, alpha=0.05, original_total=3 * R if seed == "pooled" else R)
                for metric in METRICS:
                    values = _vector(rows, dist, seed, b, method, metric) - _vector(rows, dist, seed, reference_b, reference, metric)
                    for prefix, array in (("", values), ("conditional_", values[np.isfinite(values)])):
                        for key, value in _mean_reference(mean_interval(array)).items():
                            row[f"{prefix}{metric}_{key}"] = value
                if kind == "B":
                    low, high = _vector(rows, dist, seed, 999, method, "Ase"), _vector(rows, dist, seed, 4999, method, "Ase")
                    lo_mean = float(low.mean()) if np.isfinite(low).all() else None
                    hi_mean = float(high.mean()) if np.isfinite(high).all() else None
                    ratio = hi_mean / lo_mean if lo_mean is not None and hi_mean is not None and lo_mean > 0 else None
                    row.update(mean_Ase_999=lo_mean, mean_Ase_4999=hi_mean, ratio_mean_Ase_4999_over_999=ratio)
                output.append(row)
    return output


def _reduction_reference(rows, dist, seed):
    low, high = _vector(rows, dist, seed, 999, "bca", "Ase"), _vector(rows, dist, seed, 4999, "bca", "Ase")
    values = high - 0.75 * low
    interval = mean_interval(values)
    low_mean, high_mean = (float(v.mean()) if np.isfinite(v).all() else None for v in (low, high))
    return _mean_reference(interval) | dict(status=reduction_decision(interval),
                                           conditional_interval=_mean_reference(mean_interval(values[np.isfinite(values)])),
                                           mean_Ase_999=low_mean, mean_Ase_4999=high_mean,
                                           ratio_mean_Ase_4999_over_999=high_mean / low_mean if low_mean is not None and high_mean is not None and low_mean > 0 else None)


def hypothesis_reference(rows):
    one, zero = _vector(rows, "lognormal", "pooled", 4999, "bca", "Ase"), _vector(rows, "gamma", "pooled", 4999, "bca", "Ase")
    interval = welch_interval(one, zero)
    h1 = _welch_reference(interval) | practical_decision(interval)
    h1.update(metric="Ase", margin=0.01, units="sample-SE-normalized endpoint variability, not percentage points",
              group1="lognormal", group0="gamma", independent_groups=True,
              conditional_welch=_welch_reference(welch_interval(one[np.isfinite(one)], zero[np.isfinite(zero)])))
    reductions = {dist: _reduction_reference(rows, dist, "pooled") for dist in DISTRIBUTIONS}
    seeds = [dict(master_seed=seed, H1=_welch_reference(welch_interval(_vector(rows, "lognormal", seed, 4999, "bca", "Ase"),
                                                                 _vector(rows, "gamma", seed, 4999, "bca", "Ase"))),
                  **{f"H2_{dist}": _reduction_reference(rows, dist, seed) for dist in DISTRIBUTIONS}) for seed in (0, 1, 2)]
    return dict(question_id="RQ-M06", confirmatory_intervals=3, decision_confidence=0.99, independent_originals_per_population=600,
                target_method="bca", seed_direction_gate=False, quantities=dict(H1=h1, **{f"H2_{dist}": value for dist, value in reductions.items()}),
                seed_details=seeds, hypotheses=dict(H1={key: h1[key] for key in ("status", "direction", "equivalence_supported")},
                                                   H2=dict(status=combine_reduction([value["status"] for value in reductions.values()]),
                                                           **{dist: value["status"] for dist, value in reductions.items()})))


def audit_evidence(directory):
    directory = Path(directory)
    errors, panels, rows, panel_hashes, stream_hashes = [], {}, [], {}, {}
    cells = [(dist, seed) for dist in DISTRIBUTIONS for seed in (0, 1, 2)]
    expected_panels = {f"{dist}_seed{seed}.npz" for dist, seed in cells}
    if {path.name for path in (directory / "panels").glob("*.npz")} != expected_panels:
        errors.append("saved panel files differ from six predeclared cells")
    for dist, seed in cells:
        path = directory / "panels" / f"{dist}_seed{seed}.npz"
        if not path.exists():
            continue
        panel_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with np.load(path, allow_pickle=False) as source:
                panel = {key: source[key] for key in source.files}
            errors.extend(f"{path.name}: {message}" for message in inspect_panel(panel))
            if (_meta(panel, "distribution"), _meta(panel, "master_seed"), _meta(panel, "input_accepted_head")) != (dist, seed, ACCEPTED_INPUT):
                errors.append(f"{path.name}: filename/input accepted metadata differ")
            panels[(dist, seed)] = panel
            rows.extend(original_metrics(panel))
        except (OSError, KeyError, ValueError, IndexError, TypeError) as exc:
            errors.append(f"{path.name}: unreadable/inconsistent ({type(exc).__name__})")
    expected_streams = {f"{dist}_seed{seed}_outer0_inner{inner}.npz" for dist, seed in cells for inner in (0, 1, 29)}
    if {path.name for path in (directory / "streams").glob("*.npz")} != expected_streams:
        errors.append("saved streams differ from the 18 predesignated streams")
    reconstructed = 0
    for dist, seed in cells:
        if (dist, seed) not in panels:
            continue
        for inner in (0, 1, 29):
            path = directory / "streams" / f"{dist}_seed{seed}_outer0_inner{inner}.npz"
            if not path.exists():
                continue
            stream_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                errors.extend(f"{path.name}: {message}" for message in audit_stream(path, panels[(dist, seed)], dist, seed, inner))
                reconstructed += 1
            except (OSError, KeyError, ValueError, IndexError, TypeError) as exc:
                errors.append(f"{path.name}: unreadable/inconsistent ({type(exc).__name__})")
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        errors.extend(audit_inputs(manifest["inputs"], panels))
        rng = manifest["randomness"]
        if (rng["main_namespace"] != MAIN_NAMESPACE or rng["smoke_namespace"] != 0x534D3036
                or rng["distribution_ids"] != dict(gamma=1, lognormal=2) or rng["purpose_ids"] != dict(bootstrap=1)
                or rng["spawn_key"] != ["namespace", "distribution_id", "n", "outer_id", "purpose_id", "inner_id"]
                or rng["inner_ids"] != list(range(30)) or rng["outer_ids"] != list(range(200))):
            errors.append("RNG namespace/IDs/purpose metadata differ from fixed design")
        selected = {(row["distribution"], row["master_seed"], row["outer_id"], row["inner_id"]) for row in manifest["storage"]["selected_streams"]}
        if selected != {(dist, seed, 0, inner) for dist, seed in cells for inner in (0, 1, 29)} or len(manifest["storage"]["selected_streams"]) != 18:
            errors.append("predesignated stream selection metadata differ")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        errors.append(f"manifest/input provenance unreadable/inconsistent ({type(exc).__name__})")
    seed_summary, pooled = all_summaries(rows)
    counts = {}
    for filename, expected, keys in (
        ("original_metrics.csv", rows, ("distribution", "master_seed", "outer_id", "B", "method")),
        ("summary.csv", seed_summary, ("distribution", "master_seed", "B", "method")),
        ("pooled_summary.csv", pooled, ("distribution", "B", "method")),
        ("population_contrasts.csv", population_contrasts(rows), ("master_seed", "B", "method")),
        ("paired_B_summary.csv", paired_contrasts(rows, "B"), ("distribution", "master_seed", "method", "reference", "B", "reference_B")),
        ("paired_method_summary.csv", paired_contrasts(rows, "method"), ("distribution", "master_seed", "method", "reference", "B", "reference_B")),
    ):
        try:
            saved = _csv(directory / filename)
            counts[filename] = dict(reconstructed=len(expected), saved=len(saved))
            errors.extend(compare_rows(expected, saved, keys, label=filename))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            errors.append(f"{filename}: unreadable/inconsistent ({type(exc).__name__})")
    hypotheses = hypothesis_reference(rows)
    try:
        errors.extend(compare_json(hypotheses, json.loads((directory / "hypotheses.json").read_text())))
        resource = json.loads((directory / "resources.json").read_text())
        if resource["status"] != "completed" or resource["completed_cells"] != 6 or resource["incomplete_cells"]:
            errors.append("main attempt is not complete six cells")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"hypotheses/resources unreadable/inconsistent ({type(exc).__name__})")
    if len(rows) != 7200 or reconstructed != 18:
        errors.append("reconstructed record/stream counts differ from 7200/18")
    return dict(question_id="RQ-M06", status="passed" if not errors else "failed", mismatch_count=len(errors), mismatches=errors,
                audit_implementation="independent saved-input/endpoint/selected-index-array reconstruction; no primary model or runner imports, no RNG",
                source_panels_sha256=panel_hashes, source_streams_sha256=stream_hashes, table_counts=counts,
                original_samples=sum(len(panel["outer_id"]) for panel in panels.values()), fully_reconstructed_streams=reconstructed,
                total_inner_streams=36000, hypotheses=hypotheses,
                limitations=["All 1200 selected originals, 216000 logical endpoint records, original metrics, summaries/contrasts/decisions are checked.",
                             "Exactly 18 predesignated saved index/mean streams additionally permit direct prefix/rank/linear-quantile reconstruction without RNG generation.",
                             "Other 35982 bootstrap generation sequences cannot be reconstructed from saved diagnostics; frozen-source/reference tests cover that logic.",
                             "Reused M05 originals are not a new independent replication. This internal audit is not external scientific review."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/rq_m06"))
    parser.add_argument("--output", type=Path, help="new audit JSON, preserve existing file")
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
