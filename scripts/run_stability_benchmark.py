"""Preregister and execute the single fixed RQ-M03 stability panel.

No main or smoke experiment is run by pytest. Existing outputs are preserved.
The sole development smoke uses 12 cells, R=2, K=4, B prefixes 19/49/99.
"""

from time import perf_counter
_PROCESS_START = perf_counter()

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys


class BudgetExceeded(RuntimeError):
    pass


def _deadline_handler(signum, frame):
    raise BudgetExceeded("300 second process wall budget reached")


if __name__ == "__main__" and ("--run" in sys.argv or "--smoke" in sys.argv):
    signal.signal(signal.SIGALRM, _deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 300 - (perf_counter() - _PROCESS_START)))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathscape.stability import (
    generate_sample, generate_bootstrap_means, prepare_sample, infer_prefixes,
    original_metrics, mean_t_interval, welch_interval, quantity_status,
)
from scripts.run_bootstrap_benchmark import (
    environment, sha256, peak_rss_bytes, directory_bytes, _write_json, write_csv,
)

METHODS = ("t", "percentile", "bca")
MAIN_NAMESPACE, SMOKE_NAMESPACE = 0x4D3033, 0x534D3033
DESIGN = {"distributions": ["normal", "centered_exponential"], "sample_sizes": [5, 20],
          "master_seeds": [0, 1, 2], "R": 100, "K": 20, "B_values": [199, 999, 4999],
          "alpha": .05, "true_mean": 0.0, "true_variance": 1.0, "methods": list(METHODS)}
POLICY = {"wall_seconds": 300, "peak_rss_bytes": 2 * 1024**3,
          "output_bytes": 50 * 1024**2, "max_resample_elements": 1000000,
          "decision_confidence": .99, "confirmatory_quantities": 4,
          "h2_min_pooled_group": 30, "h2_min_seed_group": 5}
SOURCES = ("mathscape/stability.py", "mathscape/bootstrap.py", "mathscape/inference.py",
           "scripts/run_stability_benchmark.py", "scripts/run_bootstrap_benchmark.py",
           "scripts/audit_stability.py", "tests/test_stability.py",
           "tests/test_stability_runner.py", "tests/test_stability_audit.py")
FIELDS = ("lower", "upper", "z0", "a", "q_low", "q_high", "tau_low", "tau_high", "valid", "reason", "tail_warning")
METRICS = {"lower_sd": "lower_sd", "upper_sd": "upper_sd", "A": "A", "coverage": "C",
           "mean_width": "mean_width", "miss_above": "miss_above", "miss_below": "miss_below", "warning_rate": "warning_rate"}


def fixed_cells():
    return [(d, n, seed) for d in DESIGN["distributions"] for n in DESIGN["sample_sizes"] for seed in DESIGN["master_seeds"]]


def sample_hash(sample):
    return hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes(order="C")).hexdigest()


def make_manifest():
    return {
        "schema_version": 1, "question_id": "RQ-M03", "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "With original samples fixed, how do finite bootstrap count and tail resolution relate to endpoint variability, and does small-sample undercoverage remain when endpoints stabilize?",
        "design": DESIGN,
        "totals": {"original_cells": 12, "original_samples": 1200, "inner_streams": 24000,
                   "B_expanded_cells": 36, "seed_summary_rows": 108, "pooled_summary_rows": 36,
                   "bootstrap_samples": 119976000, "logical_resampling_indices": 1499700000},
        "randomness": {"generator": "Generator(PCG64(SeedSequence(...)))", "entropy": "master_seed",
                       "main_namespace": MAIN_NAMESPACE, "smoke_namespace": SMOKE_NAMESPACE,
                       "distribution_ids": {"normal": 1, "centered_exponential": 2},
                       "spawn_key": ["namespace", "distribution_id", "n", "outer_id", "purpose_id", "inner_id"],
                       "purpose_ids": {"original_sample": 0, "bootstrap": 1}, "original_inner_id": 0,
                       "outer_ids": "0..99", "bootstrap_inner_ids": "0..19",
                       "excluded_from_key": ["B", "method", "display_order", "batch_size", "alpha"],
                       "prefix": "one length-4999 bootstrap mean vector per original/inner pair; B199 and B999 are prefixes, never separately generated; recompute z0 and adjusted q from each prefix",
                       "pairing": "percentile and BCa share each prefix; t and acceleration are computed once per original sample and stay unchanged over B/k",
                       "independence": "distinct keyed original and inner streams; independent original samples are the population-repetition units, not K or B; seed IDs do not replace an independence explanation"},
        "methods": {"definitions": "preserve RQ-M02: half-tie z0, delete-one jackknife acceleration, linear empirical quantiles, no clipping/fallback, same BCa denominator/order validity conditions",
                    "truth_separation": "inference receives samples/bootstrap means/alpha only; truth 0 is used solely by evaluation",
                    "tail": "endpoint tau=B*min(q,1-q); either tau<10 is an interval warning, not invalidity or exclusion",
                    "maximum_B": "4999 is not truth, infinite-B answer, or an error reference"},
        "metrics": {"se": "sample SD ddof=1 / sqrt(n)",
                    "A": "sqrt((sample Var_k(lower)+sample Var_k(upper))/2)/se, ddof=1, finite-K operational metric; not RMSE or an unbiased conditional SD estimator",
                    "coverage": "per-original C_i=mean_k I(lower<=0<=upper), never coverage of an averaged-endpoint interval",
                    "miss_above": "mean_k I(lower>0)", "miss_below": "mean_k I(upper<0)",
                    "width_warning": "average widths/warnings within original first, then average over independent originals",
                    "summary_unit": "R=100 per seed, 300 pooled originals; never Wilson or sd/sqrt(R*K)",
                    "t_reference": "stored K copies are display references; interval counts/validity use only R original t intervals, endpoint SD/A are exactly zero if valid",
                    "intervals": "99% original-level mean t intervals; zero variance makes decision interval undefined; no clipping to rate range",
                    "B999_vs_B4999": "descriptive paired original-level C differences; a nonsignificant difference is not equivalence"},
        "hypotheses": {
            "H1_n5": {"population": "centered_exponential", "n": 5, "quantity": "mean_i[A_i(4999)-0.5*A_i(199)] using all K20", "threshold": 0., "direction": "less", "support": "pooled 99% upper<0 and each seed mean<0"},
            "H1_n20": {"population": "centered_exponential", "n": 20, "quantity": "mean_i[A_i(4999)-0.5*A_i(199)] using all K20", "threshold": 0., "direction": "less", "support": "pooled 99% upper<0 and each seed mean<0"},
            "H2": {"population": "centered_exponential", "n": 20, "B": 999,
                   "classify": "W=1 iff at least 5 of k0..9 have tail_warning; required classification intervals must all be valid",
                   "evaluate": "A_eval from only independent k10..19, ddof=1; never reuse classification endpoints for A_eval",
                   "quantity": "mean A_eval(W=1)-mean A_eval(W=0)", "threshold": .02, "direction": "greater",
                   "interval": "99% original-level Welch t interval with Satterthwaite df; undefined for zero SE",
                   "groups": "each pooled group >=30 originals; each seed each group >=5; insufficiency =>inconclusive, no R/K/threshold changes",
                   "support": "pooled lower>.02 and each seed difference>.02, all group-size/validity requirements satisfied",
                   "interpretation": "association of differing originals, not causal effect of warnings"},
            "H3": {"population": "centered_exponential", "n": 5, "B": 4999,
                   "quantity": "mean over 300 independent originals of C_i from K20 coverage indicators", "threshold": .93, "direction": "less",
                   "support": "pooled 99% upper<.93 and each seed mean<.93"},
            "decision": "4 quantities with 99% intervals, approximate multiplicity control, not exact simultaneous guarantee; original-level t intervals for H1/H3, Welch for H2",
            "status": "supported iff all interval/seed/validity/group requirements hold; wholly opposite interval =>not_supported; touching/straddling threshold, unmet seed conditions, invalid/incomplete or insufficient groups =>inconclusive. Preserve contrary evidence alongside missingness. H1 retains each n; either clearly not_supported makes overall H1 not_supported",
        },
        "invalid_policy": {"original": "se<=0/nonfinite or undefined method intervals are reason-labelled; any required invalid K makes that original main A/C/width/warning metrics NA",
                           "aggregate": "any missing original needed for a confirmatory comparison makes its main estimate/interval NA and decision inconclusive; do not treat invalid as noncoverage",
                           "conditional": "jointly complete originals only may have conditional_ reference estimates with explicit finite-count denominators; never silently drop invalids"},
        "saved_schema": {"panel_axes": ["original", "B", "method", "inner"], "methods": list(METHODS),
                         "sample_hash": "SHA256 over little-endian float64 C-order original vector bytes",
                         "retain": "original vectors/IDs/hashes, endpoints, z0/a/q/tau, valid/reason/warnings; no full bootstrap mean/index arrays"},
        "policy": POLICY,
        "execution": {"main": "once only; no redraw on unsupported/inconclusive/group shortage/timeout",
                      "smoke": "one separately authorized namespace smoke: 12cells,R2,K4,prefix B19/49/99;24 originals,96 inner streams,9504 bootstrap samples; no scientific decisions",
                      "correction": "only actual code bug permits at most one separately authorized rerun, retaining reason/old source/old outputs; cumulative main wall<=600s",
                      "budget_stop": "preserve complete/incomplete cells, no automatic scale changes or retries"},
        "environment": environment(), "source_files_sha256": {name: sha256(ROOT / name) for name in SOURCES},
    }


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("question_id") != "RQ-M03":
        raise ValueError("Unsupported manifest")
    if manifest.get("design") != DESIGN or manifest.get("policy") != POLICY:
        raise ValueError("Fixed design/resource policy differs")


def verify_committed_inputs(path, manifest, commit):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != commit or len(commit) != 40:
        raise ValueError("--source-commit must be the full reviewed current HEAD")
    for name in [*SOURCES, str(path.resolve().relative_to(ROOT))]:
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if content != (ROOT / name).read_bytes():
            raise ValueError(f"Input differs from committed source: {name}")
    for name in SOURCES:
        if sha256(ROOT / name) != manifest["source_files_sha256"].get(name):
            raise ValueError(f"Source differs from preregistered hash: {name}")
    if environment() != manifest["environment"]:
        raise ValueError("Environment differs from preregistration")


def enforce_budget(wall, rss, output):
    if wall >= 300 or rss >= 2 * 1024**3 or output + 65536 >= 50 * 1024**2:
        raise BudgetExceeded("wall/RSS/output budget reached; stopping without resizing/retry")


def _allocate(R, Bs, K):
    shape = (R, len(Bs), 3, K)
    return {name: (np.full(shape, "not_computed", dtype="U96") if name == "reason" else
                   np.zeros(shape, dtype=bool) if name in ("valid", "tail_warning") else
                   np.full(shape, np.nan)) for name in FIELDS}


def run_cell(distribution, n, seed, R=100, K=20, Bs=(199, 999, 4999), namespace=MAIN_NAMESPACE,
             check=lambda: None, sample_fn=None, bootstrap_fn=None):
    if (not isinstance(R, int) or isinstance(R, bool) or not 1 <= R <= 100 or
            not isinstance(K, int) or isinstance(K, bool) or not 2 <= K <= 20):
        raise ValueError("R must be 1..100 and K must be 2..20 integers")
    if len(Bs) != 3 or any(not isinstance(b, int) or isinstance(b, bool) or not 1 <= b <= 4999 for b in Bs) or list(Bs) != sorted(set(Bs)):
        raise ValueError("Bs must be three increasing integer prefixes <=4999")
    sample_fn, bootstrap_fn = sample_fn or generate_sample, bootstrap_fn or generate_bootstrap_means
    data = _allocate(R, Bs, K)
    samples, hashes, rows = np.empty((R, n)), [], []
    for outer in range(R):
        check()
        sample = sample_fn(distribution, n, master_seed=seed, outer_id=outer, namespace=namespace)
        samples[outer] = sample
        hashes.append(sample_hash(sample))
        context = prepare_sample(sample, alpha=.05)
        se = float(context["se"])
        for inner in range(K):
            check()
            numerical_failure = ""
            if not context["valid"]:
                # Preserve scientific undefined cases without attempting a
                # resampling call that rejects the original sample itself.
                means = np.full(max(Bs), np.nan)
            else:
                try:
                    means = bootstrap_fn(sample, distribution, n, master_seed=seed, outer_id=outer,
                                         inner_id=inner, Bmax=max(Bs), namespace=namespace,
                                         max_elements=1000000)
                except ValueError as exc:
                    if str(exc) != "bootstrap_mean_nonfinite":
                        raise  # configuration errors/bugs must stop the run.
                    numerical_failure = str(exc)
                    means = np.full(max(Bs), np.nan)
            records = infer_prefixes(context, means, B_values=Bs)
            if numerical_failure:
                for B in Bs:
                    for method in ("percentile", "bca"):
                        records[B][method]["reason"] = numerical_failure
            for bi, B in enumerate(Bs):
                for mi, method in enumerate(METHODS):
                    for name in FIELDS:
                        data[name][outer, bi, mi, inner] = records[B][method][name]
        outer_data = {name: values[outer] for name, values in data.items()}
        metrics = original_metrics(outer_data, se)
        h2 = {"h2_class_complete": False, "h2_eval_complete": False,
              "h2_class_warning_count": float("nan"), "h2_W": float("nan"), "h2_A_eval": float("nan")}
        if 999 in Bs and K == 20:
            bi = list(Bs).index(999)
            h2["h2_class_complete"] = bool(data["valid"][outer, bi, 2, :10].all())
            if h2["h2_class_complete"]:
                h2["h2_class_warning_count"] = int(data["tail_warning"][outer, bi, 2, :10].sum())
                h2["h2_W"] = int(h2["h2_class_warning_count"] >= 5)
            last10 = {name: values[outer, bi:bi+1, :, 10:] for name, values in data.items()}
            eval_metrics = original_metrics(last10, se)
            h2["h2_eval_complete"] = bool(eval_metrics["complete"][0, 2])
            h2["h2_A_eval"] = float(eval_metrics["A"][0, 2])
        for bi, B in enumerate(Bs):
            for mi, method in enumerate(METHODS):
                row = dict(distribution=distribution, n=n, master_seed=seed, outer_id=outer,
                           B=B, method=method, R=R, K=K, alpha=.05, se=se, sample_sha256=hashes[-1])
                for key, values in metrics.items():
                    row[key] = values[bi, mi].item()
                row.update(h2 if method == "bca" and B == 999 else {key: False if isinstance(value, bool) else float("nan") for key, value in h2.items()})
                rows.append(row)
    check()
    return data, samples, np.array(hashes), rows


def _metric_stats(values):
    values = np.asarray(values, dtype=float)
    result = mean_t_interval(values, confidence=.99)
    return {"mean": result["mean"], "sd": result["sd"], "mcse": result["mcse"],
            "decision_low": result["low"], "decision_high": result["high"],
            "finite_count": result["finite_count"]}


def summarize_rows(originals, pooled=False):
    rows = []
    for distribution in DESIGN["distributions"]:
        for n in DESIGN["sample_sizes"]:
            seeds = ["pooled"] if pooled else DESIGN["master_seeds"]
            for seed in seeds:
                for B in DESIGN["B_values"]:
                    for method in METHODS:
                        group = [r for r in originals if r["distribution"] == distribution and r["n"] == n and r["B"] == B and r["method"] == method and (pooled or r["master_seed"] == seed)]
                        complete = [r for r in group if r["complete"]]
                        expected = 300 if pooled else 100
                        row = dict(distribution=distribution, n=n, master_seed=seed,
                                   seed_group="0,1,2" if pooled else str(seed), B=B, method=method,
                                   R=300 if pooled else 100, K=20, alpha=.05,
                                   original_total=expected, observed_original_count=len(group),
                                   missing_original_count=expected-len(group), complete_original_count=len(complete),
                                   incomplete_original_count=expected-len(complete),
                                   interval_count=sum(r["total_inner"] for r in group),
                                   valid_count=sum(r["valid_inner"] for r in group),
                                   invalid_count=sum(r["invalid_inner"] for r in group),
                                   conditional_original_count=len(complete))
                        for name, source in METRICS.items():
                            main_values = [r[source] if r["complete"] else float("nan") for r in group]
                            if len(group) != expected:
                                main_values += [float("nan")] * max(1, expected-len(group))
                            main = _metric_stats(main_values)
                            conditional = _metric_stats([r[source] for r in complete])
                            for suffix in ("mean", "sd", "mcse", "decision_low", "decision_high"):
                                key = name if suffix == "mean" else f"{name}_{suffix}"
                                row[key] = main[suffix]
                                row[f"conditional_{key}"] = conditional[suffix]
                        rows.append(row)
    return rows


def _find(originals, distribution, n, seed, B, method):
    return sorted([r for r in originals if r["distribution"] == distribution and r["n"] == n and r["master_seed"] == seed and r["B"] == B and r["method"] == method], key=lambda r: r["outer_id"])


def _quantity(values, seed_values, threshold, direction):
    interval = mean_t_interval(values, confidence=.99)
    points = [mean_t_interval(v, confidence=.99)["mean"] for v in seed_values]
    status = quantity_status(interval["low"], interval["high"], threshold, direction, points)
    complete_design = (len(values) == 300 and len(seed_values) == 3 and
                       all(len(v) == 100 for v in seed_values) and np.isfinite(values).all())
    conditional = mean_t_interval(np.asarray(values)[np.isfinite(values)], confidence=.99)
    if not complete_design:
        interval.update(mean=float("nan"), sd=float("nan"), mcse=float("nan"),
                        low=float("nan"), high=float("nan"), valid=False,
                        reason="invalid/incomplete original panel; require 100 valid IDs per seed and 300 pooled")
        status = "inconclusive"
    return {**interval, "threshold": threshold, "direction": direction, "seed_points": points,
            "status": status, "complete_design": complete_design,
            "conditional_interval": conditional, "independent_unit": "original sample"}


def _expected_values(rows, field):
    """Preserve every predeclared original ID, representing missing rows as NA."""
    by_id = {r["outer_id"]: r for r in rows}
    if len(by_id) != len(rows) or any(i not in range(100) for i in by_id):
        raise ValueError("Duplicate or out-of-range original IDs")
    return np.array([by_id[i][field] if i in by_id else float("nan") for i in range(100)])


def assemble_hypotheses(originals):
    quantities = {}
    for n in (5, 20):
        seed_values = []
        for seed in (0, 1, 2):
            lowB, highB = _find(originals, "centered_exponential", n, seed, 199, "bca"), _find(originals, "centered_exponential", n, seed, 4999, "bca")
            seed_values.append(_expected_values(highB, "A")-.5*_expected_values(lowB, "A"))
        quantities[f"H1_n{n}"] = _quantity(np.concatenate(seed_values), seed_values, 0., "less")
    h3_seeds = [_expected_values(_find(originals, "centered_exponential", 5, seed, 4999, "bca"), "C") for seed in (0, 1, 2)]
    quantities["H3"] = _quantity(np.concatenate(h3_seeds), h3_seeds, .93, "less")
    h2_rows = [r for r in originals if r["distribution"] == "centered_exponential" and r["n"] == 20 and r["B"] == 999 and r["method"] == "bca"]
    group_rows, seed_differences, seed_group_counts = [], [], []
    pooled_groups = None
    expected_keys = {(seed, outer) for seed in (0, 1, 2) for outer in range(100)}
    actual_keys = {(r["master_seed"], r["outer_id"]) for r in h2_rows}
    if len(actual_keys) != len(h2_rows) or not actual_keys <= expected_keys:
        raise ValueError("Duplicate/out-of-range original IDs in H2")
    all_complete = actual_keys == expected_keys and all(r["h2_class_complete"] and r["h2_eval_complete"] for r in h2_rows)
    for seed in (0, 1, 2, "pooled"):
        cohort = h2_rows if seed == "pooled" else [r for r in h2_rows if r["master_seed"] == seed]
        groups = [np.array([r["h2_A_eval"] for r in cohort if r["h2_class_complete"] and r["h2_eval_complete"] and r["h2_W"] == W]) for W in (0, 1)]
        expected_count = 300 if seed == "pooled" else 100
        cohort_complete = len(cohort) == expected_count and all(r["h2_class_complete"] and r["h2_eval_complete"] for r in cohort)
        for W in (0, 1):
            stats = mean_t_interval(groups[W], confidence=.99)
            group_rows.append(dict(master_seed=seed, W=W, original_total=expected_count,
                                   observed_original_count=len(cohort), missing_original_count=expected_count-len(cohort),
                                   group_count=len(groups[W]), all_required_complete=cohort_complete,
                                   unclassified_original_count=sum(not r["h2_class_complete"] for r in cohort),
                                   invalid_evaluation_count=sum(not r["h2_eval_complete"] for r in cohort),
                                   A_eval_mean=stats["mean"] if cohort_complete else float("nan"),
                                   A_eval_sd=stats["sd"] if cohort_complete else float("nan"),
                                   A_eval_mcse=stats["mcse"] if cohort_complete else float("nan"),
                                   conditional_A_eval_mean=stats["mean"], conditional_A_eval_sd=stats["sd"],
                                   conditional_A_eval_mcse=stats["mcse"]))
        comparison = welch_interval(groups[1], groups[0], confidence=.99,
                                    min_group=30 if seed == "pooled" else 5)
        if seed == "pooled":
            pooled_groups = comparison
        else:
            seed_differences.append(comparison["difference"])
            seed_group_counts.append([len(groups[0]), len(groups[1])])
    enough = pooled_groups["N0"] >= 30 and pooled_groups["N1"] >= 30 and all(min(pair) >= 5 for pair in seed_group_counts)
    h2_status = quantity_status(pooled_groups["low"], pooled_groups["high"], .02, "greater", seed_differences)
    quantities["H2"] = {**{key: value if all_complete or key not in ("mean1", "mean0", "variance1", "variance0", "difference", "se", "df", "low", "high") else float("nan") for key, value in pooled_groups.items()},
                        "conditional_welch": pooled_groups, "threshold": .02, "direction": "greater",
                        "seed_points": seed_differences, "seed_group_counts_W0_W1": seed_group_counts,
                        "all_required_complete": all_complete, "group_requirements_met": enough,
                        "valid": bool(pooled_groups["valid"] and all_complete and enough),
                        "reason": ("invalid/incomplete required original panel" if not all_complete else
                                   pooled_groups["reason"] or ("insufficient_seed_group_count" if not enough else "")),
                        "original_total": 300,
                        "complete_original_count": sum(r["h2_class_complete"] and r["h2_eval_complete"] for r in h2_rows),
                        "conditional_interval_status": h2_status,
                        "status": h2_status if all_complete and enough else "inconclusive",
                        "interpretation": "association, not causal effect; class k0..9, evaluation k10..19"}
    h1_statuses = [quantities[f"H1_n{n}"]["status"] for n in (5, 20)]
    h1 = "not_supported" if "not_supported" in h1_statuses else "supported" if h1_statuses == ["supported", "supported"] else "inconclusive"
    return {"question_id": "RQ-M03", "confirmatory_quantities": 4, "decision_confidence": .99,
            "pooled_independent_originals": 300, "quantities": quantities,
            "hypotheses": {"H1": {"status": h1, "n5": h1_statuses[0], "n20": h1_statuses[1]},
                           "H2": {"status": quantities["H2"]["status"]}, "H3": {"status": quantities["H3"]["status"]}}}, group_rows


def coverage_differences(originals):
    rows = []
    for dist in DESIGN["distributions"]:
        for n in DESIGN["sample_sizes"]:
            for method in METHODS:
                seed_values = []
                for seed in (0, 1, 2):
                    low, high = _find(originals, dist, n, seed, 999, method), _find(originals, dist, n, seed, 4999, method)
                    if [r["outer_id"] for r in low] != [r["outer_id"] for r in high]:
                        raise ValueError("Unpaired original IDs in descriptive coverage difference")
                    values = np.array([hi["C"]-lo["C"] for lo, hi in zip(low, high)])
                    seed_values.append(values)
                    rows.append(dict(distribution=dist, n=n, method=method, master_seed=seed,
                                     description="C4999-C999; descriptive, not equivalence", **mean_t_interval(values, confidence=.99)))
                rows.append(dict(distribution=dist, n=n, method=method, master_seed="pooled",
                                 description="C4999-C999; descriptive, not equivalence", **mean_t_interval(np.concatenate(seed_values), confidence=.99)))
    return rows


def write_figure(pooled, output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    output.mkdir(exist_ok=False)
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Normal: endpoint variability A", "Centered exponential: endpoint variability A", "Normal: coverage", "Centered exponential: coverage"])
    colors = {"t": "#007c83", "percentile": "#a84624", "bca": "#6d4ab1"}
    for col, dist in enumerate(DESIGN["distributions"], 1):
        for n in (5, 20):
            for method in METHODS:
                group = sorted([r for r in pooled if r["distribution"] == dist and r["n"] == n and r["method"] == method], key=lambda r: r["B"])
                for row, metric in ((1, "A"), (2, "coverage")):
                    fig.add_trace(go.Scatter(x=[str(r["B"]) for r in group], y=[r[metric] for r in group],
                                             mode="lines+markers", name=f"{method}, n={n}", legendgroup=f"{method}-{n}",
                                             showlegend=row == col == 1, line=dict(color=colors[method], dash="solid" if n == 5 else "dash"),
                                             marker=dict(symbol="circle" if n == 5 else "square"),
                                             error_y=dict(type="data", symmetric=False,
                                                          array=[r[f"{metric}_decision_high"]-r[metric] for r in group],
                                                          arrayminus=[r[metric]-r[f"{metric}_decision_low"] for r in group])), row=row, col=col)
        fig.add_hline(y=.95, line_dash="dot", row=2, col=col)
    fig.update_xaxes(type="category", title_text="Bootstrap count B (same-stream prefixes)")
    fig.update_yaxes(title_text="Mean A over 300 originals", row=1)
    fig.update_yaxes(title_text="Mean C over 300 originals", row=2)
    fig.update_layout(template="plotly_white", width=1250, height=950,
                      title="RQ-M03: stability vs coverage; 300 originals, K=20; bars=99% original-level t intervals", margin=dict(t=110,b=120), legend=dict(orientation="h", y=-.15))
    fig.write_html(str(output / "overview.html"), include_plotlyjs=True, full_html=True, config={"displaylogo": False})


def _final_resources(output, record):
    record.update(process_wall_seconds=perf_counter()-_PROCESS_START, process_peak_rss_bytes=peak_rss_bytes(), output_bytes_before_resources=directory_bytes(output))
    record["output_bytes_total"] = record["output_bytes_before_resources"]
    for _ in range(5):
        # Reuse the JSON-safe conversion through the same public writer format.
        total = record["output_bytes_before_resources"] + len((json.dumps(record, indent=2, allow_nan=False)+"\n").encode())
        if total == record["output_bytes_total"]:
            break
        record["output_bytes_total"] = total
    _write_json(output / "resources.json", record)


def run_benchmark(manifest_path, output, source_commit):
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    verify_committed_inputs(manifest_path, manifest, source_commit)
    output = output.resolve()
    if ROOT not in output.parents:
        raise ValueError("Output must be inside project")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("attempt.json", "resources.json", "panels", "summary.csv", "original_metrics.csv", "figures")):
        raise FileExistsError("Existing attempt/results preserved")
    attempt = dict(question_id="RQ-M03", status="started", started_utc=datetime.now(timezone.utc).isoformat(),
                   source_git_sha=source_commit, manifest_sha256=sha256(manifest_path), namespace=MAIN_NAMESPACE, single_main_attempt=True)
    _write_json(output / "attempt.json", attempt)
    (output / "panels").mkdir()
    originals, completed = [], []
    resources = {**attempt, "environment": environment(), "R": 100, "K": 20, "B_values": [199,999,4999],
                 "total_cells": 12, "bootstrap_arrays_saved": False, "original_samples_saved": True,
                 "max_resample_elements": 1000000,
                 "wall_scope": "module start before heavy imports through calculation/outputs, excluding interpreter startup and final resource write; external time records full CLI"}
    def check():
        enforce_budget(perf_counter()-_PROCESS_START, peak_rss_bytes(), directory_bytes(output))
    try:
        for dist, n, seed in fixed_cells():
            resources["active_cell"] = [dist,n,seed]
            data, samples, hashes, rows = run_cell(dist,n,seed,check=check)
            path = output / "panels" / f"{dist}_n{n}_seed{seed}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, **data, samples=samples, sample_sha256=hashes, outer_id=np.arange(100),
                                    B_values=np.array([199,999,4999]), methods=np.array(METHODS), inner_id=np.arange(20),
                                    distribution=dist,n=n,master_seed=seed,R=100,K=20,alpha=.05,namespace=MAIN_NAMESPACE)
            originals.extend(rows)
            completed.append([dist,n,seed])
            check()
        write_csv(output / "original_metrics.csv", originals)
        summary, pooled = summarize_rows(originals), summarize_rows(originals, pooled=True)
        write_csv(output / "summary.csv", summary)
        write_csv(output / "pooled_summary.csv", pooled)
        hypotheses, groups = assemble_hypotheses(originals)
        write_csv(output / "H2_groups.csv", groups)
        _write_json(output / "hypotheses.json", hypotheses)
        write_csv(output / "coverage_difference.csv", coverage_differences(originals))
        write_figure(pooled, output / "figures")
        check()
        resources["status"] = "completed"
    except Exception as exc:
        resources.update(status="stopped", stop_reason=f"{type(exc).__name__}: {exc}")
        if originals and not (output / "original_metrics.csv").exists():
            write_csv(output / "original_metrics.csv", originals)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        resources.update(completed_cells=len(completed), completed_cell_keys=completed,
                         incomplete_cells=[list(c) for c in fixed_cells() if list(c) not in completed], finished_utc=datetime.now(timezone.utc).isoformat())
        _final_resources(output, resources)
    return resources


def run_smoke(output):
    output = output.resolve()
    if ROOT not in output.parents:
        raise ValueError("Smoke output must be inside project")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Existing smoke attempt preserved")
    attempt = dict(question_id="RQ-M03", scope="smoke only, no scientific decisions", R=2,K=4,B_values=[19,49,99],namespace=SMOKE_NAMESPACE,
                   source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})
    _write_json(output / "attempt.json", attempt)
    result, complete = {**attempt,"environment":environment(),"status":"started"}, []
    try:
        for dist,n,seed in fixed_cells():
            data, samples, hashes, rows = run_cell(dist,n,seed,R=2,K=4,Bs=(19,49,99),namespace=SMOKE_NAMESPACE,
                                                   check=lambda: enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output)))
            complete.append(dict(distribution=dist,n=n,master_seed=seed,valid_count=int(data["valid"].sum())))
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="stopped",stop_reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(completed_cells=len(complete),cells=complete,original_samples=2*len(complete),inner_streams=8*len(complete))
        _final_resources(output,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-manifest",action="store_true")
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--smoke",action="store_true")
    parser.add_argument("--manifest",type=Path,default=ROOT/"evidence/rq_m03/manifest.json")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source-commit")
    args=parser.parse_args()
    if args.prepare_manifest:
        path=args.manifest.resolve()
        if ROOT not in path.parents:
            parser.error("Manifest must be inside project")
        path.parent.mkdir(parents=True,exist_ok=True)
        _write_json(path,make_manifest())
        print("Prepared",path.relative_to(ROOT))
    elif args.smoke:
        result=run_smoke(args.output or ROOT/"evidence/rq_m03/smoke")
        print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))
    else:
        if not args.source_commit:
            parser.error("--run requires reviewed full --source-commit")
        result=run_benchmark(args.manifest.resolve(),args.output or ROOT/"evidence/rq_m03",args.source_commit)
        print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))


if __name__=="__main__":
    main()
