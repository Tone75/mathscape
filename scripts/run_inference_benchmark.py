"""Preregister and run the single fixed RQ-M01 experiment.

Normal pytest never runs this fixed experiment. Prepare the manifest, review
and commit it together with the engine/runner, then explicitly invoke --run.
Existing attempts/results are never overwritten and no automatic retry occurs.
"""

from __future__ import annotations

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
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathscape.inference import (  # noqa: E402
    normal_power, rng_for_cell, sample_batch, summarize, t_inference,
)

DESIGN = {
    "distributions": ["normal", "centered_exponential"],
    "sample_sizes": [5, 20, 100],
    "effects": [0.0, 0.2, 0.5],
    "master_seeds": [0, 1, 2],
    "B": 5000, "mu0": 0.0, "sigma": 1.0, "alpha": 0.05, "trial_id": 0,
}
SOURCE_FILES = ("mathscape/inference.py", "scripts/run_inference_benchmark.py")
POLICY = {"max_batch": 1000, "wall_seconds": 300,
          "peak_rss_bytes": 2 * 1024**3, "output_bytes": 50 * 1024**2,
          "normal_comparisons": 54, "familywise_epsilon": 0.001}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": version("numpy"),
            "scipy": version("scipy"), "os": platform.system(),
            "os_release": platform.release(), "architecture": platform.machine()}


def make_manifest(root: Path = ROOT) -> dict:
    """Return the preregistration before observing any fixed-cell results."""
    return {
        "schema_version": 1, "question_id": "RQ-M01",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "How do sample size and population shape change one-sample t inference?",
        "design": DESIGN,
        "totals": {"cells": 54, "sample_groups": 270000, "observations": 11250000},
        "randomness": {
            "generator": "numpy.random.Generator(numpy.random.PCG64(SeedSequence(...)))",
            "distribution_ids": {"normal": 1, "centered_exponential": 2},
            "entropy": "master_seed",
            "spawn_key": ["distribution_id", "n", "d_float64_low32", "d_float64_high32", "trial_id"],
            "effect_encoding": "IEEE754 float64 little-endian uint64; -0.0 canonicalized to +0.0; low32=bits&0xffffffff; high32=bits>>32",
            "consumption": "one generator per cell; sequential row-major generation across batches; alpha and B are not in the RNG key",
            "generation": {"normal": "mu0 + sigma * (d + Z), Z~N(0,1)",
                           "centered_exponential": "mu0 + sigma * (d + E - 1), E~Exp(scale=1)"},
            "independence": "independent pseudo-random draws within and between sample groups; fixed seed IDs provide reproducibility, not a proof of independence",
        },
        "inference": {"variance_ddof": 1, "test": "two-sided one-sample t",
                      "reject": "unrounded p < alpha", "coverage": "closed interval contains true mean mu0+sigma*d",
                      "boundary": "CI/test equivalence checks exclude isclose(abs(t), critical, rtol=1e-12, atol=1e-12); reported p and intervals are never adjusted",
                      "invalid": "retain B denominator and valid-only counts; any invalid group marks cell for investigation and aggregate rates/widths/intervals as NaN",
                      "rate_intervals": "95% Wilson, descriptive per cell; not simultaneous acceptance bands",
                      "width_sd": "sample SD ddof=1 across B widths", "width_mcse": "width_sd/sqrt(B)",
                      "rate_mcse": "sqrt(rate*(1-rate)/B)",
                      "normal_theory": "nct.cdf(-critical,n-1,sqrt(n)*d)+nct.sf(critical,n-1,sqrt(n)*d); not assigned to nonnormal rows"},
        "policy": POLICY,
        "normal_investigation_band": {
            "comparisons": 54, "familywise_epsilon": 0.001,
            "tail_probability": "0.001/(2*54)",
            "integer_bounds": "binom.ppf(tail,B,p), binom.ppf(1-tail,B,p), inclusive",
            "p": "coverage=0.95; rejection=normal noncentral-t power (including alpha under d=0)",
            "interpretation": "conservative investigation flag, not a proof/disproof and never a reason to redraw",
        },
        "hypotheses": {
            "H1": {"claim": "Normal iid t intervals cover at nominal level; under H0 rejection matches alpha",
                   "criterion": "supported if all 27 normal coverage counts and all 9 normal d=0 rejection counts lie in preregistered conservative bands, with zero invalids; otherwise unresolved/investigate, retaining every result"},
            "H2": {"claim": "For normal populations, increasing n narrows mean CI width and increases theoretical power at nonzero fixed d",
                   "criterion": "supported if mean widths strictly decrease for n=5,20,100 within each normal d/seed, normal theory power strictly increases at d=0.2,0.5, every normal empirical rejection count is in its conservative theory band, and no invalids; otherwise unresolved/investigate; no single-sample monotonicity claim"},
            "H3": {"claim": "Matched mean/variance need not give nominal small-sample inference under a right-skewed population",
                   "scope": "n=5,d=0; larger n and nonzero effects reported descriptively, without a required recovery direction",
                   "criterion": "supported if coverage or rejection for centered_exponential lies outside its nominal conservative binomial count range in all three seeds, with the same direction across seeds, while all matched normal counts lie inside their nominal ranges and all six cells have zero invalids; otherwise unresolved/not supported by this fixed design",
                   "nominal_band": "same conservative count construction p=0.95 or0.05, tail=0.001/(2*54); descriptive comparator only, excluded from the 54 normal-theory checks",
                   "direction": "neither direction nor n=100 recovery is fixed as a desired conclusion"},
        },
        "execution": {"first_fixed_run": "once only; no automatic repeats, seed additions or outcome-dependent filtering",
                      "bug_correction": "at most one separately authorized rerun; retain old results, source version and reason; this runner refuses any existing attempt in a destination",
                      "samples": "raw observations are never saved; at most one batch plus B inference outputs per cell retained",
                      "budget_response": "stop, preserve completed rows and record incomplete cells/resources; do not expand limits"},
        "environment": environment(),
        "source_files_sha256": {name: sha256(root / name) for name in SOURCE_FILES},
    }


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("question_id") != "RQ-M01":
        raise ValueError("Unsupported manifest")
    if manifest.get("design") != DESIGN or manifest.get("policy") != POLICY:
        raise ValueError("Fixed design or resource policy differs from preregistration")
    if manifest.get("totals") != {"cells": 54, "sample_groups": 270000, "observations": 11250000}:
        raise ValueError("Fixed experiment totals differ")


def fixed_cells():
    return list(itertools.product(DESIGN["distributions"], DESIGN["sample_sizes"],
                                  DESIGN["effects"], DESIGN["master_seeds"]))


def conservative_count_band(B: int, probability: float) -> tuple[int, int]:
    tail = POLICY["familywise_epsilon"] / (2 * POLICY["normal_comparisons"])
    return int(binom.ppf(tail, B, probability)), int(binom.ppf(1 - tail, B, probability))


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class BudgetExceeded(RuntimeError):
    pass


def enforce_budget(wall_seconds: float, rss_bytes: int, output_bytes: int) -> None:
    if wall_seconds >= POLICY["wall_seconds"]:
        raise BudgetExceeded("wall time budget reached")
    if rss_bytes >= POLICY["peak_rss_bytes"]:
        raise BudgetExceeded("process peak RSS budget reached")
    if output_bytes >= POLICY["output_bytes"]:
        raise BudgetExceeded("output size budget reached")


def verify_committed_inputs(manifest_path: Path, manifest: dict, source_commit: str, root: Path = ROOT) -> None:
    """Bind the run to a reviewed, committed manifest and engine/runner."""
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != source_commit or len(source_commit) != 40:
        raise ValueError("--source-commit must be the full current HEAD SHA")
    for name in [*SOURCE_FILES, str(manifest_path.resolve().relative_to(root.resolve()))]:
        committed = subprocess.check_output(["git", "show", f"{source_commit}:{name}"], cwd=root)
        if committed != (root / name).read_bytes():
            raise ValueError(f"Input is not identical to committed version: {name}")
    for name in SOURCE_FILES:
        if sha256(root / name) != manifest["source_files_sha256"].get(name):
            raise ValueError(f"Source hash differs from preregistration: {name}")
    if environment() != manifest["environment"]:
        raise ValueError("Runtime environment differs from preregistration")


def run_cell(distribution: str, n: int, d: float, master_seed: int,
             check_budget=lambda: None, B: int = 5000, batch_size: int = 1000) -> dict:
    """Compute one cell in bounded batches. Tiny B is used only by unit tests."""
    if not isinstance(B, int) or isinstance(B, bool) or not 2 <= B <= 5000:
        raise ValueError("B must be an integer 2..5000")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be an integer 1..1000")
    rng = rng_for_cell(distribution, n, d, master_seed, trial_id=0)
    accumulated = {}
    for offset in range(0, B, batch_size):
        check_budget()
        batch = sample_batch(rng, distribution, n, min(batch_size, B - offset),
                             d=d, mu0=0.0, sigma=1.0)
        result = t_inference(batch, mu0=0.0, alpha=0.05)
        for name, values in result.items():
            accumulated.setdefault(name, []).append(values)
    check_budget()
    result = {name: np.concatenate(values) for name, values in accumulated.items()}
    summary = summarize(result, true_mean=d)
    theory = normal_power(n, d, alpha=0.05) if distribution == "normal" else float("nan")
    row = {"distribution": distribution, "n": n, "d": d, "master_seed": master_seed,
           "trial_id": 0, "alpha": 0.05, "B": B, **summary,
           "theory_power": theory,
           "theory_diff": summary["reject_rate"] - theory}
    for metric, count, probability in [("coverage", summary["cover_count"], 0.95),
                                        ("reject", summary["reject_count"], theory)]:
        if distribution == "normal":
            low, high = conservative_count_band(B, probability)
            row[f"{metric}_count_band_low"] = low
            row[f"{metric}_count_band_high"] = high
            row[f"{metric}_investigate"] = bool(summary["invalid_count"] or not low <= count <= high)
        else:
            row[f"{metric}_count_band_low"] = float("nan")
            row[f"{metric}_count_band_high"] = float("nan")
            row[f"{metric}_investigate"] = "not_applicable"
    row["invalid_investigate"] = bool(summary["invalid_count"])
    return row


def _svg_panel(title, rows, metric, target=None, show_normal_theory=False) -> str:
    """Small static figure: all three seeds retained, no outcome selection."""
    width, height = 960, 420
    left, top, chart_w, chart_h = 90, 65, 790, 275
    values = [float(r[metric]) for r in rows if np.isfinite(float(r[metric]))]
    if show_normal_theory:
        values.extend(float(r["theory_power"]) for r in rows if r["distribution"] == "normal")
    if target is not None:
        values.append(target)
    lower = min(values) if values else 0
    upper = max(values) if values else 1
    padding = max((upper - lower) * 0.15, 0.005)
    lower, upper = max(0, lower - padding), upper + padding
    if metric in ("coverage", "reject_rate"):
        upper = min(1, upper)
    def yp(value):
        return top + chart_h * (upper - value) / (upper - lower)
    colors = {"normal": "#007c83", "centered_exponential": "#a84624"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#fcfcf8"/>',
             '<g font-family="sans-serif" fill="#1d292e">',
             f'<text x="35" y="30" font-size="18">{title}</text>']
    for i in range(6):
        value = lower + (upper - lower) * i / 5
        y = yp(value)
        parts.extend([f'<line x1="{left}" x2="{left+chart_w}" y1="{y}" y2="{y}" stroke="#d9e0df"/>',
                      f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-size="12">{value:.3f}</text>'])
    if target is not None:
        y = yp(target)
        parts.append(f'<line x1="{left}" x2="{left+chart_w}" y1="{y}" y2="{y}" stroke="#555" stroke-dasharray="5,5"/>')
    if show_normal_theory:
        points = []
        for i, n in enumerate([5, 20, 100]):
            theory = next(r["theory_power"] for r in rows if r["distribution"] == "normal" and int(r["n"]) == n)
            points.append(f"{left+chart_w*(i+.5)/3},{yp(theory)}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#007c83" stroke-width="1.5" stroke-dasharray="5,5"/>')
    for i, n in enumerate([5, 20, 100]):
        x = left + chart_w * (i + 0.5) / 3
        parts.append(f'<text x="{x}" y="{top+chart_h+23}" text-anchor="middle" font-size="14">n={n}</text>')
        for row in rows:
            if int(row["n"]) != n:
                continue
            value = float(row[metric])
            if not np.isfinite(value):
                continue
            dx = (-12 if row["distribution"] == "normal" else 12) + (int(row["master_seed"]) - 1) * 5
            color = colors[row["distribution"]]
            parts.append(f'<circle cx="{x+dx}" cy="{yp(value)}" r="4" fill="{color}"/>')
    if show_normal_theory:
        parts.append('<text x="90" y="367" font-size="11">Dashed teal: exact normal-population noncentral-t power; no nonnormal theory curve.</text>')
    parts.extend(['<text x="90" y="389" font-size="12" fill="#007c83">Normal (all seeds 0,1,2)</text>',
                  '<text x="335" y="389" font-size="12" fill="#a84624">Centered exponential (all seeds 0,1,2)</text>',
                  '<text x="90" y="409" font-size="11">Fixed B=5000 per cell; point estimates only. CSV includes Wilson intervals and Monte Carlo errors.</text>',
                  '</g></svg>'])
    return "\n".join(parts)


def write_figures(rows: list[dict], output: Path) -> None:
    output.mkdir(exist_ok=False)
    h0 = [r for r in rows if r["d"] == 0]
    (output / "h0_coverage.svg").write_text(_svg_panel("RQ-M01: H0 coverage (nominal 0.95 dashed)", h0, "coverage", .95), encoding="utf-8")
    (output / "h0_rejection.svg").write_text(_svg_panel("RQ-M01: H0 rejection (nominal 0.05 dashed)", h0, "reject_rate", .05), encoding="utf-8")
    effect = [r for r in rows if r["d"] == .5]
    (output / "d05_power.svg").write_text(_svg_panel("RQ-M01: observed rejection and normal theory at d=0.5", effect, "reject_rate", show_normal_theory=True), encoding="utf-8")
    (output / "h0_mean_width.svg").write_text(_svg_panel("RQ-M01: mean confidence-interval width at d=0 (standardized units)", h0, "mean_width"), encoding="utf-8")


def run_benchmark(manifest_path: Path, output: Path, source_commit: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    verify_committed_inputs(manifest_path, manifest, source_commit)
    output = output.resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError("Output must be a dedicated directory inside this project")
    output.mkdir(parents=True, exist_ok=True)
    occupied = [name for name in ("attempt.json", "summary.csv", "resources.json", "figures") if (output / name).exists()]
    if occupied:
        raise FileExistsError("Existing attempt/results are preserved; refusing overwrite: " + ", ".join(occupied))
    attempt = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
               "source_git_sha": source_commit, "manifest_sha256": sha256(manifest_path),
               "policy": "single explicit attempt; no automatic rerun"}
    with (output / "attempt.json").open("x", encoding="utf-8") as handle:
        json.dump(attempt, handle, indent=2)
        handle.write("\n")
    start = perf_counter()
    completed = []
    resources = {**attempt, "environment": environment(), "completed_cells": 0,
                 "total_cells": 54, "batch_size": 1000, "raw_samples_saved": False,
                 "wall_scope": "benchmark batches, CSV and figures; excludes imports/preflight and final resources write",
                 "rss_scope": "whole runner process lifetime; Darwin ru_maxrss is bytes, other Unix KiB converted to bytes"}
    def check():
        enforce_budget(perf_counter() - start, peak_rss_bytes(), directory_bytes(output))
    try:
        check()
        with (output / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = None
            for distribution, n, d, seed in fixed_cells():
                row = run_cell(distribution, n, d, seed, check_budget=check)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                completed.append(row)
                check()
        write_figures(completed, output / "figures")
        check()
        resources["status"] = "completed"
    except Exception as exc:
        resources["status"] = "stopped"
        resources["stop_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        completed_keys = {(r["distribution"], r["n"], r["d"], r["master_seed"]) for r in completed}
        resources.update({"completed_cells": len(completed),
                          "incomplete_cells": [list(c) for c in fixed_cells() if c not in completed_keys],
                          "wall_seconds": perf_counter() - start,
                          "process_peak_rss_bytes": peak_rss_bytes(),
                          "output_bytes_before_resources": directory_bytes(output),
                          "finished_utc": datetime.now(timezone.utc).isoformat()})
        # Include this JSON itself in the final output size without writing it
        # twice. Its decimal size field converges after at most a few updates.
        resources["output_bytes_total"] = resources["output_bytes_before_resources"]
        for _ in range(5):
            payload = json.dumps(resources, indent=2, allow_nan=False) + "\n"
            total = resources["output_bytes_before_resources"] + len(payload.encode("utf-8"))
            if total == resources["output_bytes_total"]:
                break
            resources["output_bytes_total"] = total
        with (output / "resources.json").open("x", encoding="utf-8") as handle:
            json.dump(resources, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-manifest", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evidence/rq_m01/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/rq_m01")
    parser.add_argument("--source-commit", help="full reviewed HEAD SHA; required for --run")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    if args.prepare_manifest:
        if ROOT.resolve() not in args.manifest.resolve().parents:
            parser.error("Manifest must be inside the project")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x", encoding="utf-8") as handle:
            json.dump(make_manifest(), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print("Prepared", args.manifest.relative_to(ROOT))
    else:
        if not args.source_commit:
            parser.error("--run requires --source-commit (full reviewed HEAD SHA)")
        result = run_benchmark(args.manifest, args.output, args.source_commit)
        print(json.dumps({name: result[name] for name in ("status", "completed_cells", "wall_seconds", "process_peak_rss_bytes")}))


if __name__ == "__main__":
    main()
