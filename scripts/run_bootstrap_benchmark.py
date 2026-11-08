"""Preregister and execute the one authorized RQ-M02 bootstrap comparison.

The fixed main experiment is never run by pytest. --smoke is a distinct,
single recorded 18-cell R=2/B_boot=99 check in the reserved smoke namespace.
Existing attempts, endpoints and summaries are never overwritten.
"""

from time import perf_counter
_PROCESS_START = perf_counter()

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys


class BudgetExceeded(RuntimeError):
    """Stop without silently changing the fixed experiment."""


def _deadline_handler(signum, frame):
    raise BudgetExceeded("300 second process wall budget reached")


# Arm before heavy imports for the actual CLI experiment, never on import by
# pytest or during manifest preparation. The outer tool also measures full CLI.
if __name__ == "__main__" and ("--run" in sys.argv or "--smoke" in sys.argv):
    signal.signal(signal.SIGALRM, _deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, 300 - (perf_counter() - _PROCESS_START)))

import numpy as np
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathscape.bootstrap import draw_replicate, intervals
from mathscape.inference import wilson_interval

METHODS = ("t", "percentile", "bca")
PAIRS = (("percentile", "t"), ("bca", "percentile"), ("bca", "t"))
MAIN_NAMESPACE = 0x4D3032
SMOKE_NAMESPACE = 0x534D3032
DESIGN = {"distributions": ["normal", "centered_exponential"], "sample_sizes": [5, 20, 100],
          "master_seeds": [0, 1, 2], "R": 2000, "B_boot": 999, "alpha": .05,
          "true_mean": 0.0, "true_variance": 1.0, "d": 0.0, "methods": list(METHODS)}
POLICY = {"wall_seconds": 300, "peak_rss_bytes": 2 * 1024**3,
          "output_bytes": 50 * 1024**2, "max_resample_elements": 1000000,
          "normal_t_comparisons": 9, "normal_t_familywise_epsilon": .001,
          "decision_confidence": .99, "decision_quantities": 5}
SOURCES = ("mathscape/bootstrap.py", "mathscape/inference.py", "scripts/run_bootstrap_benchmark.py")
Z99 = float(norm.ppf(.995))


def environment():
    return {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy"),
            "plotly": version("plotly"), "os": platform.system(), "os_release": platform.release(),
            "architecture": platform.machine()}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_cells():
    return list(itertools.product(DESIGN["distributions"], DESIGN["sample_sizes"], DESIGN["master_seeds"]))


def make_manifest():
    return {
        "schema_version": 1, "question_id": "RQ-M02", "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "For matched populations, how do t, percentile and BCa mean intervals change coverage, width and miss direction on the same samples?",
        "design": DESIGN, "totals": {"generation_cells": 18, "outer_sample_groups": 36000,
                                      "method_summary_rows": 54, "bootstrap_sample_groups": 35964000,
                                      "logical_resampling_indices": 1498500000},
        "randomness": {
            "generator": "numpy.random.Generator(numpy.random.PCG64(numpy.random.SeedSequence(...)))",
            "main_namespace": MAIN_NAMESPACE, "smoke_namespace": SMOKE_NAMESPACE,
            "entropy": "master_seed", "distribution_ids": {"normal": 1, "centered_exponential": 2},
            "spawn_key": ["namespace", "distribution_id", "n", "replicate_id", "purpose_id"],
            "purposes": {"original_sample": 0, "bootstrap_resampling": 1},
            "excluded_from_key": ["method", "alpha", "display_order", "batch_size"],
            "replicate_ids": "0..1999 inclusive, each with its own original and resampling streams",
            "pairing": "all 3 methods share the same original sample; percentile and BCa share exactly the same bootstrap mean vector",
            "independence": "distinct outer replicates have independently keyed original and resampling streams; no common index array is broadcast across replicates; seed IDs alone are not proof of independence",
        },
        "methods": {
            "t": "existing unknown-variance ddof=1, df=n-1 two-sided t interval",
            "percentile": "quantile(bootstrap_means,[alpha/2,1-alpha/2],method='linear')",
            "bca": {"ties": "(# bootstrap means < sample mean + 0.5 * # equal)/B_boot",
                    "z0": "normal.ppf(tie_adjusted_probability)",
                    "acceleration": "sum((x-mean(x))**3)/(6*sum((x-mean(x))**2)**1.5), independently checked against delete-one jackknife",
                    "adjusted_quantile": "Phi(z0+(z0+z_p)/(1-a*(z0+z_p)))",
                    "no_fallback": "no clipping or substitution with t/percentile when BCa is undefined"},
            "finite_bootstrap": "evaluate the finite B_boot=999 procedure, not B_boot->infinity or a separate inner-MC convergence study",
        },
        "invalid_policy": {
            "reasons": "nonfinite sample, n<2, constant sample, nonfinite z0/a, bad adjusted quantile order and other undefined numerical outputs remain reason-labelled invalid",
            "main_aggregates": "any invalid method interval makes its unconditional rates and width summaries NA; total R and valid/invalid counts remain explicit",
            "pair_aggregates": "any invalid paired interval makes main paired differences NA; joint-valid reference values are explicitly prefixed conditional_ and never used for decisions",
            "tail_warning": "B_boot*min(q,1-q)<10 at either endpoint; recorded separately, does not invalidate or change the method",
        },
        "summaries": {
            "coverage": "closed interval includes true mean 0",
            "miss_above": "lower > true mean (interval wholly above truth)",
            "miss_below": "upper < true mean (interval wholly below truth)",
            "partition": "for fully valid cells coverage+miss_above+miss_below=1",
            "descriptive_rates": "95% Wilson intervals and Bernoulli MCSE",
            "decision_rates": "99% Wilson intervals, columns separate from descriptive 95% intervals",
            "width": "mean, median, sample SD ddof=1, MCSE=SD/sqrt(R), 99% mean +/- z(.995)*MCSE",
            "paired": "per-outer-replicate coverage indicator and width differences, sample SD ddof=1 / sqrt(R); preserve sums and sums of squares",
            "pooling": "sum sufficient statistics over the 3 seed cells (6000 independent outer replicates), never treat 3 seed means as sample size 3; pooled median uses saved interval widths only",
        },
        "decision": {
            "scope": "hypotheses refer only to centered_exponential; normal is reference, n=100 descriptive",
            "confidence": "each of 5 preregistered quantities uses a 99% interval, a Bonferroni-type approximate familywise 95% decision; paired normal approximation is not an exact simultaneous guarantee",
            "quantities": {
                "H1_width": {"n": 5, "comparison": "percentile-t", "quantity": "mean_width_difference", "desired": "upper < 0", "threshold": 0.0, "direction": "less"},
                "H1_coverage": {"n": 5, "comparison": "percentile-t", "quantity": "coverage_difference", "desired": "upper < -0.02", "threshold": -.02, "direction": "less"},
                "H2_improvement": {"n": 20, "comparison": "bca-percentile", "quantity": "coverage_difference", "desired": "lower > 0.01", "threshold": .01, "direction": "greater"},
                "H2_level": {"n": 20, "method": "bca", "quantity": "coverage", "desired": "upper < 0.96", "threshold": .96, "direction": "less"},
                "H3_level": {"n": 5, "method": "bca", "quantity": "coverage", "desired": "upper < 0.93", "threshold": .93, "direction": "less"},
            },
            "hypothesis_quantities": {"H1": ["H1_width", "H1_coverage"], "H2": ["H2_improvement", "H2_level"], "H3": ["H3_level"]},
            "support": "all relevant pooled 99% intervals satisfy the strict desired inequalities and every seed point estimate is strictly on the same desired side of its threshold",
            "not_supported": "at least one relevant interval lies completely on the opposite side (strictly beyond threshold); record this contrary evidence before discussing mixed seeds",
            "inconclusive": "interval touches/straddles threshold, mixed seed directions, any invalid or incomplete cell; contrary intervals are separately recorded",
            "normal_t_gate": "all 9 normal t coverage counts in inclusive binom.ppf(epsilon/(2*9),R,.95)..binom.ppf(1-epsilon/(2*9),R,.95), epsilon=.001; unresolved reference checks prevent a final scientific conclusion",
            "prohibitions": "no threshold changes, seed selection, method additions, extra R/B/n or redraws in response to outcomes",
        },
        "policy": POLICY,
        "execution": {"main": "once only; bug correction at most one separately authorized attempt retaining reason, old source and old outputs, cumulative main wall <=600 seconds",
                      "smoke": "one separately authorized R=2,B_boot=99 check over 18 cells (36 outer samples) in smoke namespace; never used to choose hypotheses/design",
                      "over_budget": "save completed and incomplete cells, stop; no automatic resizing, expansion or rerun",
                      "retained": "endpoints, valid/reason/tail flags only; never raw samples or full bootstrap arrays"},
        "environment": environment(), "source_files_sha256": {name: sha256(ROOT / name) for name in SOURCES},
    }


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("question_id") != "RQ-M02":
        raise ValueError("Unsupported manifest")
    if manifest.get("design") != DESIGN or manifest.get("policy") != POLICY:
        raise ValueError("Fixed design/resource policy differs")
    if manifest.get("randomness", {}).get("main_namespace") != MAIN_NAMESPACE:
        raise ValueError("Main namespace differs")


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def enforce_budget(wall, rss, output):
    if wall >= POLICY["wall_seconds"]:
        raise BudgetExceeded("Process wall budget reached")
    if rss >= POLICY["peak_rss_bytes"]:
        raise BudgetExceeded("Process peak RSS budget reached")
    # Leave a bounded reserve for the final stop/resource record.
    if output + 65536 >= POLICY["output_bytes"]:
        raise BudgetExceeded("Output budget reached (resource-record reserve included)")


def verify_committed_inputs(manifest_path, manifest, source_commit):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if len(source_commit) != 40 or source_commit != actual:
        raise ValueError("--source-commit must be the full current reviewed HEAD SHA")
    relative_manifest = str(manifest_path.resolve().relative_to(ROOT))
    for name in [*SOURCES, relative_manifest]:
        committed = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=ROOT)
        if committed != (ROOT / name).read_bytes():
            raise ValueError(f"Input differs from committed source: {name}")
    for name in SOURCES:
        if sha256(ROOT / name) != manifest["source_files_sha256"].get(name):
            raise ValueError(f"Source hash differs: {name}")
    if environment() != manifest["environment"]:
        raise ValueError("Environment differs from preregistration")


def _width_stats(values):
    values = np.asarray(values, dtype=float)
    count = len(values)
    mean = float(np.mean(values)) if count else float("nan")
    sd = float(np.std(values, ddof=1)) if count > 1 else float("nan")
    se = sd / np.sqrt(count) if count else float("nan")
    return {"mean_width": mean, "width_sd": sd,
            "width_median": float(np.median(values)) if count else float("nan"),
            "width_mcse": float(se), "mean_width_decision_low": mean - Z99 * se,
            "mean_width_decision_high": mean + Z99 * se}


def method_summary(lower, upper, valid, tail_warning):
    lower, upper, valid = np.asarray(lower), np.asarray(upper), np.asarray(valid, dtype=bool)
    total, count = len(valid), int(valid.sum())
    cover = valid & (lower <= 0) & (upper >= 0)
    above, below = valid & (lower > 0), valid & (upper < 0)
    assert int(cover.sum() + above.sum() + below.sum()) == count
    result = {"total": total, "valid_count": count, "invalid_count": total - count,
              "cover_count": int(cover.sum()), "miss_above_count": int(above.sum()),
              "miss_below_count": int(below.sum()), "tail_warning_count": int(np.sum(tail_warning & valid))}
    for prefix, numerator in [("coverage", int(cover.sum())), ("miss_above", int(above.sum())), ("miss_below", int(below.sum()))]:
        rate = numerator / count if count else float("nan")
        low, high = wilson_interval(numerator, count) if count else (float("nan"), float("nan"))
        result[f"conditional_{prefix}"] = rate
        result[f"conditional_{prefix}_low"] = low
        result[f"conditional_{prefix}_high"] = high
        result[prefix] = rate if count == total else float("nan")
        result[f"{prefix}_low"] = low if count == total else float("nan")
        result[f"{prefix}_high"] = high if count == total else float("nan")
        result[f"{prefix}_mcse"] = float(np.sqrt(rate * (1 - rate) / count)) if count == total else float("nan")
    decision = wilson_interval(int(cover.sum()), total, confidence=.99) if count == total else (float("nan"), float("nan"))
    result["coverage_decision_low"], result["coverage_decision_high"] = decision
    conditional = _width_stats((upper - lower)[valid])
    result.update({f"conditional_{key}": value for key, value in conditional.items()})
    result.update({key: value if count == total else float("nan") for key, value in conditional.items()})
    result["width_sum"] = float(np.sum((upper - lower)[valid]))
    result["width_sumsq"] = float(np.sum((upper - lower)[valid] ** 2))
    return result


def paired_summary(data, method, reference):
    a, b = METHODS.index(method), METHODS.index(reference)
    valid = data["valid"][a] & data["valid"][b]
    ca = ((data["lower"][a] <= 0) & (data["upper"][a] >= 0)).astype(int)
    cb = ((data["lower"][b] <= 0) & (data["upper"][b] >= 0)).astype(int)
    coverage = (ca - cb)[valid]
    width = ((data["upper"][a] - data["lower"][a]) - (data["upper"][b] - data["lower"][b]))[valid]
    return pair_from_sufficient(total=len(valid), valid_count=int(valid.sum()),
                                coverage_sum=float(coverage.sum()), coverage_sumsq=float((coverage ** 2).sum()),
                                width_sum=float(width.sum()), width_sumsq=float((width ** 2).sum()))


def pair_from_sufficient(total, valid_count, coverage_sum, coverage_sumsq, width_sum, width_sumsq):
    result = {"total": int(total), "pair_valid_count": int(valid_count), "pair_invalid_count": int(total - valid_count),
              "coverage_diff_sum": coverage_sum, "coverage_diff_sumsq": coverage_sumsq,
              "width_diff_sum": width_sum, "width_diff_sumsq": width_sumsq}
    for prefix, summed, squared in [("coverage_diff", coverage_sum, coverage_sumsq), ("width_diff", width_sum, width_sumsq)]:
        mean = summed / valid_count if valid_count else float("nan")
        variance = max(0.0, (squared - summed * summed / valid_count) / (valid_count - 1)) if valid_count > 1 else float("nan")
        sd = np.sqrt(variance)
        se = sd / np.sqrt(valid_count) if valid_count else float("nan")
        stats = {prefix: mean, f"{prefix}_sd": float(sd), f"{prefix}_mcse": float(se),
                 f"{prefix}_decision_low": mean - Z99 * se, f"{prefix}_decision_high": mean + Z99 * se}
        result.update({f"conditional_{key}": value for key, value in stats.items()})
        result.update({key: value if valid_count == total else float("nan") for key, value in stats.items()})
    return result


def pool_pairs(rows):
    return pair_from_sufficient(sum(int(r["total"]) for r in rows), sum(int(r["pair_valid_count"]) for r in rows),
                                sum(r["coverage_diff_sum"] for r in rows), sum(r["coverage_diff_sumsq"] for r in rows),
                                sum(r["width_diff_sum"] for r in rows), sum(r["width_diff_sumsq"] for r in rows))


def run_cell(distribution, n, seed, R=2000, B_boot=999, namespace=MAIN_NAMESPACE, check=lambda: None, draw=None):
    if not isinstance(R, int) or isinstance(R, bool) or not 1 <= R <= 2000:
        raise ValueError("R must be integer 1..2000")
    if not isinstance(B_boot, int) or isinstance(B_boot, bool) or not 1 <= B_boot <= 999:
        raise ValueError("B_boot must be integer 1..999")
    draw = draw or draw_replicate
    data = {"lower": np.full((3, R), np.nan), "upper": np.full((3, R), np.nan),
            "valid": np.zeros((3, R), dtype=bool), "reason": np.full((3, R), "not_computed", dtype="U128"),
            "tail_warning": np.zeros((3, R), dtype=bool), "q_low": np.full((3, R), np.nan),
            "q_high": np.full((3, R), np.nan), "tail_count_low": np.full((3, R), np.nan),
            "tail_count_high": np.full((3, R), np.nan)}
    for replicate in range(R):
        check()
        sample, means = draw(distribution, n, master_seed=seed, replicate_id=replicate, B_boot=B_boot, namespace=namespace)
        records = intervals(sample, means, alpha=.05)
        for index, method in enumerate(METHODS):
            record = records[method]
            for key in data:
                data[key][index, replicate] = record[key]
    check()
    metadata = {"distribution": distribution, "n": n, "master_seed": seed, "R": R, "B_boot": B_boot, "alpha": .05}
    summaries = [{**metadata, "method": method,
                  **method_summary(data["lower"][i], data["upper"][i], data["valid"][i], data["tail_warning"][i])}
                 for i, method in enumerate(METHODS)]
    pairs = [{**metadata, "method": method, "reference": reference,
              **paired_summary(data, method, reference)} for method, reference in PAIRS]
    return data, summaries, pairs


def pooled_outputs(cell_data, summaries, pairs):
    pooled, pooled_pairs = [], []
    for distribution, n in itertools.product(DESIGN["distributions"], DESIGN["sample_sizes"]):
        data = {key: np.concatenate([cell_data[(distribution, n, seed)][key] for seed in DESIGN["master_seeds"]], axis=1)
                for key in ("lower", "upper", "valid", "tail_warning")}
        meta = {"distribution": distribution, "n": n, "seed_group": "0,1,2", "R": 6000, "B_boot": 999, "alpha": .05}
        for index, method in enumerate(METHODS):
            pooled.append({**meta, "method": method,
                           **method_summary(data["lower"][index], data["upper"][index], data["valid"][index], data["tail_warning"][index])})
        for method, reference in PAIRS:
            selected = [r for r in pairs if r["distribution"] == distribution and r["n"] == n and r["method"] == method and r["reference"] == reference]
            pooled_pairs.append({**meta, "method": method, "reference": reference, **pool_pairs(selected)})
    return pooled, pooled_pairs


def normal_t_band(R=2000):
    tail = .001 / (2 * 9)
    return int(binom.ppf(tail, R, .95)), int(binom.ppf(1 - tail, R, .95))


def quantity_status(lower, upper, threshold, direction, seed_points):
    if not all(np.isfinite([lower, upper, *seed_points])):
        return "inconclusive"
    if direction == "less":
        if lower > threshold:
            return "not_supported"
        return "supported" if upper < threshold and all(p < threshold for p in seed_points) else "inconclusive"
    if upper < threshold:
        return "not_supported"
    return "supported" if lower > threshold and all(p > threshold for p in seed_points) else "inconclusive"


def decide_hypotheses(summaries, pairs, pooled, pooled_pairs):
    lookup = lambda rows, n, method, reference=None: next(r for r in rows if r["distribution"] == "centered_exponential" and r["n"] == n and r["method"] == method and (reference is None or r["reference"] == reference))
    quantities = {}
    specs = [("H1_width", 5, "percentile", "t", "width_diff", 0., "less"),
             ("H1_coverage", 5, "percentile", "t", "coverage_diff", -.02, "less"),
             ("H2_improvement", 20, "bca", "percentile", "coverage_diff", .01, "greater"),
             ("H2_level", 20, "bca", None, "coverage", .96, "less"),
             ("H3_level", 5, "bca", None, "coverage", .93, "less")]
    for name, n, method, reference, metric, threshold, direction in specs:
        pooled_row = lookup(pooled_pairs if reference else pooled, n, method, reference)
        seed_rows = [r for r in (pairs if reference else summaries) if r["distribution"] == "centered_exponential" and r["n"] == n and r["method"] == method and (reference is None or r["reference"] == reference)]
        points = [r[metric] for r in sorted(seed_rows, key=lambda r: r["master_seed"])]
        low, high = pooled_row[f"{metric}_decision_low"], pooled_row[f"{metric}_decision_high"]
        quantities[name] = {"estimate": pooled_row[metric], "decision_low": low, "decision_high": high,
                            "threshold": threshold, "direction": direction, "seed_points": points,
                            "criterion_status": quantity_status(low, high, threshold, direction, points)}
    low, high = normal_t_band()
    normal_rows = [r for r in summaries if r["distribution"] == "normal" and r["method"] == "t"]
    normal_checks = [{"n": r["n"], "master_seed": r["master_seed"], "cover_count": r["cover_count"],
                      "invalid_count": r["invalid_count"], "band_low": low, "band_high": high,
                      "passed": r["invalid_count"] == 0 and low <= r["cover_count"] <= high} for r in normal_rows]
    gate = len(normal_checks) == 9 and all(r["passed"] for r in normal_checks)
    complete_valid = len(summaries) == 54 and all(r["invalid_count"] == 0 for r in summaries)
    hypotheses = {}
    for name, names in {"H1": ["H1_width", "H1_coverage"], "H2": ["H2_improvement", "H2_level"], "H3": ["H3_level"]}.items():
        statuses = [quantities[q]["criterion_status"] for q in names]
        status = "not_supported" if "not_supported" in statuses else ("supported" if all(s == "supported" for s in statuses) else "inconclusive")
        hypotheses[name] = {"quantities": names, "criterion_status": status,
                            "final_status": status if gate and complete_valid else "inconclusive",
                            "gate_note": "normal t reference resolved and all cells valid" if gate and complete_valid else "normal t reference unresolved or invalid/incomplete cells; no final scientific conclusion"}
    return {"question_id": "RQ-M02", "decision_confidence": .99, "pooled_outer_replicates": 6000,
            "normal_t_reference_resolved": gate, "all_cells_complete_and_valid": complete_valid,
            "normal_t_checks": normal_checks, "quantities": quantities, "hypotheses": hypotheses}


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(pooled, output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    output.mkdir(exist_ok=False)
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Normal: coverage (99% Wilson)", "Centered exponential: coverage (99% Wilson)", "Normal: mean width (99% MC interval)", "Centered exponential: mean width (99% MC interval)"])
    for col, distribution in enumerate(DESIGN["distributions"], 1):
        for method, color in zip(METHODS, ["#007c83", "#a84624", "#6d4ab1"]):
            rows = sorted([r for r in pooled if r["distribution"] == distribution and r["method"] == method], key=lambda r: r["n"])
            for row, metric, low, high in [(1, "coverage", "coverage_decision_low", "coverage_decision_high"), (2, "mean_width", "mean_width_decision_low", "mean_width_decision_high")]:
                values = [r[metric] for r in rows]
                fig.add_trace(go.Scatter(x=[str(r["n"]) for r in rows], y=values, mode="lines+markers", name=method,
                                         legendgroup=method, showlegend=col == row == 1, line_color=color,
                                         error_y=dict(type="data", symmetric=False, array=[r[high]-r[metric] for r in rows], arrayminus=[r[metric]-r[low] for r in rows])), row=row, col=col)
        fig.add_hline(y=.95, line_dash="dot", row=1, col=col)
    fig.update_xaxes(type="category", title_text="Observations per outer sample n")
    fig.update_yaxes(title_text="Coverage", row=1)
    fig.update_yaxes(title_text="Mean interval width", row=2)
    fig.update_layout(template="plotly_white", width=1150, height=850,
                      title="RQ-M02: all seeds pooled (6000 outer samples); B_boot=999, alpha=.05", margin=dict(t=100))
    fig.write_html(str(output / "overview.html"), include_plotlyjs=True, full_html=True, config={"displaylogo": False})


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _final_resource_record(output, resources):
    resources["process_wall_seconds"] = perf_counter() - _PROCESS_START
    resources["process_peak_rss_bytes"] = peak_rss_bytes()
    resources["output_bytes_before_resources"] = directory_bytes(output)
    resources["output_bytes_total"] = resources["output_bytes_before_resources"]
    for _ in range(5):
        payload = json.dumps(_json_safe(resources), indent=2, allow_nan=False) + "\n"
        total = resources["output_bytes_before_resources"] + len(payload.encode())
        if total == resources["output_bytes_total"]:
            break
        resources["output_bytes_total"] = total
    _write_json(output / "resources.json", resources)


def run_benchmark(manifest_path, output, source_commit):
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    verify_committed_inputs(manifest_path, manifest, source_commit)
    output = output.resolve()
    if ROOT not in output.parents:
        raise ValueError("Output must be inside the project")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["attempt.json", "resources.json", "summary.csv", "intervals", "figures"]):
        raise FileExistsError("Existing attempt/results preserved; refusing overwrite")
    attempt = {"question_id": "RQ-M02", "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
               "source_git_sha": source_commit, "manifest_sha256": sha256(manifest_path), "namespace": MAIN_NAMESPACE,
               "single_main_attempt": True}
    _write_json(output / "attempt.json", attempt)
    (output / "intervals").mkdir()
    summaries, pairs, cell_data, completed = [], [], {}, []
    resources = {**attempt, "environment": environment(), "completed_cells": 0, "total_cells": 18,
                 "raw_samples_saved": False, "bootstrap_arrays_saved": False,
                 "wall_scope": "module start before heavy imports through computation/output, excluding interpreter startup and final resources write; outer tool separately measures full CLI",
                 "max_resample_elements": 1000000, "R": 2000, "B_boot": 999}
    def check():
        enforce_budget(perf_counter() - _PROCESS_START, peak_rss_bytes(), directory_bytes(output))
    try:
        for distribution, n, seed in fixed_cells():
            resources["active_cell"] = [distribution, n, seed]
            check()
            data, method_rows, pair_rows = run_cell(distribution, n, seed, check=check)
            stem = f"{distribution}_n{n}_seed{seed}"
            np.savez_compressed(output / "intervals" / f"{stem}.npz", **data,
                                methods=np.array(METHODS), replicate_id=np.arange(2000),
                                distribution=distribution, n=n, master_seed=seed, alpha=.05, R=2000, B_boot=999, namespace=MAIN_NAMESPACE)
            cell_data[(distribution, n, seed)] = data
            summaries.extend(method_rows)
            pairs.extend(pair_rows)
            completed.append([distribution, n, seed])
            check()
        write_csv(output / "summary.csv", summaries)
        write_csv(output / "paired_summary.csv", pairs)
        pooled, pooled_pairs = pooled_outputs(cell_data, summaries, pairs)
        write_csv(output / "pooled_summary.csv", pooled)
        write_csv(output / "pooled_paired_summary.csv", pooled_pairs)
        _write_json(output / "hypotheses.json", decide_hypotheses(summaries, pairs, pooled, pooled_pairs))
        write_figures(pooled, output / "figures")
        check()
        resources["status"] = "completed"
    except Exception as exc:
        resources["status"] = "stopped"
        resources["stop_reason"] = f"{type(exc).__name__}: {exc}"
        if summaries and not (output / "summary.csv").exists():
            write_csv(output / "summary.csv", summaries)
        if pairs and not (output / "paired_summary.csv").exists():
            write_csv(output / "paired_summary.csv", pairs)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        resources.update(completed_cells=len(completed), completed_cell_keys=completed,
                         incomplete_cells=[list(cell) for cell in fixed_cells() if list(cell) not in completed],
                         finished_utc=datetime.now(timezone.utc).isoformat())
        _final_resource_record(output, resources)
    return resources


def run_smoke(output):
    output = output.resolve()
    if ROOT not in output.parents:
        raise ValueError("Smoke output must be inside project")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Existing smoke attempt preserved")
    attempt = {"question_id": "RQ-M02", "scope": "smoke only; no scientific decisions", "R": 2, "B_boot": 99,
               "namespace": SMOKE_NAMESPACE, "outer_samples_max": 36,
               "source_hashes": {name: sha256(ROOT / name) for name in SOURCES}}
    _write_json(output / "attempt.json", attempt)
    completed = []
    result = {**attempt, "environment": environment(), "status": "started"}
    try:
        for distribution, n, seed in fixed_cells():
            data, _, _ = run_cell(distribution, n, seed, R=2, B_boot=99, namespace=SMOKE_NAMESPACE,
                                  check=lambda: enforce_budget(perf_counter()-_PROCESS_START, peak_rss_bytes(), directory_bytes(output)))
            completed.append({"distribution": distribution, "n": n, "master_seed": seed,
                              "valid_counts": data["valid"].sum(axis=1).tolist()})
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "stopped"
        result["stop_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(completed_cells=len(completed), cells=completed, completed_outer_groups=2*len(completed))
        _final_resource_record(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-manifest", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evidence/rq_m02/manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.prepare_manifest:
        path = args.manifest.resolve()
        if ROOT not in path.parents:
            parser.error("Manifest must be inside project")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, make_manifest())
        print("Prepared", path.relative_to(ROOT))
    elif args.smoke:
        result = run_smoke(args.output or ROOT / "evidence/rq_m02/smoke")
        print(json.dumps({name: result[name] for name in ["status", "completed_cells", "process_wall_seconds", "process_peak_rss_bytes"]}))
    else:
        if not args.source_commit:
            parser.error("--run requires full reviewed --source-commit")
        result = run_benchmark(args.manifest.resolve(), args.output or ROOT / "evidence/rq_m02", args.source_commit)
        print(json.dumps({name: result[name] for name in ["status", "completed_cells", "process_wall_seconds", "process_peak_rss_bytes"]}))


if __name__ == "__main__":
    main()
