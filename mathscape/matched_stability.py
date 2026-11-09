"""RQ-M06: verified saved originals and conditional endpoint variation.

This module never generates original observations. Its only random operation
resamples an explicitly supplied, unchanged original sample. Inference reuses
the accepted t/percentile/BCa implementation without population information.
"""

from hashlib import sha256
from io import BytesIO
import json
from numbers import Integral, Real
from pathlib import Path
import subprocess

import numpy as np

from mathscape import stability
from mathscape.matched_moments import practical_difference_status

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_HEAD = "dc2501c3099f5c25227d9eee00d227c2f0b02f81"
M05_MANIFEST_PATH = "evidence/rq_m05/manifest.json"
DISTRIBUTIONS = ("gamma", "lognormal")
DISTRIBUTION_IDS = {"gamma": 1, "lognormal": 2}
METHODS = ("t", "percentile", "bca")
B_VALUES = (999, 4999)
N = 20
MAX_OUTER = 200
MAX_INNER = 30
MAX_B = 4999
MAX_ELEMENTS = 1_000_000
MAX_ID = 2**32 - 1
MAIN_NAMESPACE = 0x4D3036
SMOKE_NAMESPACE = 0x534D3036
M05_NAMESPACE = 0x4D3035
INPUT_PATHS = (M05_MANIFEST_PATH,) + tuple(
    f"evidence/rq_m05/panels/{distribution}_seed{seed}.npz"
    for distribution in DISTRIBUTIONS for seed in (0, 1, 2)
)
prepare_sample = stability.prepare_sample
mean_t_interval = stability.mean_t_interval
welch_interval = stability.welch_interval


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _distribution(distribution):
    if not isinstance(distribution, str) or distribution not in DISTRIBUTION_IDS:
        raise ValueError("distribution must be gamma or lognormal")
    return distribution


def sample_sha256(sample):
    """Hash the specified float64 little-endian C-order observation bytes."""
    return sha256(np.asarray(sample, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def _accepted_blob(root, relative_path):
    """Read accepted Git content without changing the index or worktree."""
    try:
        return subprocess.run(
            ["git", "show", f"{ACCEPTED_HEAD}:{relative_path}"], cwd=root,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ValueError(f"accepted input unavailable: {relative_path}") from exc


def _verified_blob(root, relative_path):
    try:
        local = (root / relative_path).read_bytes()
    except OSError as exc:
        raise ValueError(f"saved input unavailable: {relative_path}") from exc
    accepted = _accepted_blob(root, relative_path)
    if local != accepted:
        raise ValueError(f"saved input differs from accepted Git blob: {relative_path}")
    return local


def _scalar(panel, key, expected):
    if key not in panel or panel[key].shape != () or panel[key].item() != expected:
        raise ValueError(f"M05 metadata mismatch: {key}")


def _inspect_inputs(root):
    """Verify all seven accepted files; select IDs without using endpoints."""
    root = Path(root).resolve()
    source_files = {}
    blobs = {}
    for path in INPUT_PATHS:
        blob = _verified_blob(root, path)
        blobs[path] = blob
        source_files[path] = sha256(blob).hexdigest()
    try:
        source_manifest = json.loads(blobs[M05_MANIFEST_PATH])
    except (ValueError, UnicodeError) as exc:
        raise ValueError("M05 manifest is invalid JSON") from exc
    design = source_manifest.get("design", {})
    expected_design = {"distributions": list(DISTRIBUTIONS), "n": N,
                       "master_seeds": [0, 1, 2], "R": 10000, "K": 1,
                       "B": MAX_B, "alpha": .05, "methods": list(METHODS)}
    if source_manifest.get("question_id") != "RQ-M05" or any(
            design.get(key) != value for key, value in expected_design.items()):
        raise ValueError("M05 manifest design does not match accepted input contract")
    cells, selections = {}, []
    for distribution in DISTRIBUTIONS:
        for seed in (0, 1, 2):
            path = f"evidence/rq_m05/panels/{distribution}_seed{seed}.npz"
            try:
                with np.load(BytesIO(blobs[path]), allow_pickle=False) as panel:
                    for key, expected in {"distribution": distribution, "master_seed": seed,
                            "n": N, "R": 10000, "observed_original_count": 10000,
                            "K": 1, "B": MAX_B, "alpha": .05,
                            "namespace": M05_NAMESPACE}.items():
                        _scalar(panel, key, expected)
                    if not np.array_equal(panel["methods"], np.asarray(METHODS)):
                        raise ValueError("M05 method metadata mismatch")
                    samples = panel["samples"]
                    ids, hashes = panel["outer_id"], panel["sample_sha256"]
                    if samples.shape != (10000, N) or samples.dtype != np.dtype("<f8") or not np.isfinite(samples).all():
                        raise ValueError("M05 sample array shape/dtype/finite mismatch")
                    if ids.dtype != np.dtype("uint16") or not np.array_equal(ids, np.arange(10000)):
                        raise ValueError("M05 original ID mapping mismatch")
                    if hashes.shape != (10000,) or hashes.dtype != np.dtype("S64"):
                        raise ValueError("M05 saved sample hashes have wrong shape/dtype")
                    calculated = np.asarray([sample_sha256(row).encode("ascii") for row in samples], dtype="S64")
                    if not np.array_equal(calculated, hashes):
                        raise ValueError("M05 saved original hash mismatch")
                    selected = {"samples": samples[:MAX_OUTER].copy(),
                                "outer_id": ids[:MAX_OUTER].copy(),
                                "sample_sha256": hashes[:MAX_OUTER].copy()}
            except (OSError, KeyError, EOFError) as exc:
                raise ValueError(f"M05 panel cannot be read: {path}") from exc
            for array in selected.values():
                array.flags.writeable = False
            cells[(distribution, seed)] = selected
            selections.append({"distribution": distribution, "master_seed": seed,
                               "source_path": path, "outer_ids": list(range(MAX_OUTER)),
                               "sample_sha256": [h.decode("ascii") for h in selected["sample_sha256"]],
                               "selected_samples_sha256": sample_sha256(selected["samples"])})
    manifest = {"schema_version": 1, "question_id": "RQ-M06", "accepted_head": ACCEPTED_HEAD,
                "source_question_id": "RQ-M05", "source_files_sha256": source_files,
                "selection_rule": "each gamma/lognormal x seed0,1,2; exact saved outer IDs0..199; no endpoint-based selection",
                "n": N, "originals_per_cell": MAX_OUTER, "selected_originals": 1200,
                "sample_hash_contract": "SHA256 little-endian float64 C-order bytes",
                "selections": selections}
    return manifest, cells


def make_inputs_manifest(root=ROOT):
    """Build deterministic provenance; read only, never write or draw samples."""
    return _inspect_inputs(root)[0]


def validate_inputs_manifest(inputs, root=ROOT):
    """Reject file drift, accepted-blob drift, hash drift or changed selection."""
    expected, _ = _inspect_inputs(root)
    if not isinstance(inputs, dict) or inputs != expected:
        raise ValueError("M06 inputs manifest does not match verified accepted originals")


def load_inputs(inputs=None, root=ROOT):
    """Return six verified cells keyed by (distribution, seed), read-only arrays.

    A supplied manifest must exactly match the fixed selection and current
    accepted bytes. Endpoints are not consulted and no original generator is
    present. Each cell has samples(200,20), outer_id(200), sample_sha256(200).
    """
    expected, cells = _inspect_inputs(root)
    if inputs is not None and (not isinstance(inputs, dict) or inputs != expected):
        raise ValueError("M06 inputs manifest does not match verified accepted originals")
    return cells


def rng_for_panel(distribution, n=N, master_seed=0, outer_id=0, purpose_id=1,
                  inner_id=0, namespace=MAIN_NAMESPACE):
    """New bootstrap keys only; purpose0 original generation is forbidden."""
    distribution = _distribution(distribution)
    n = _integer(n, "n", N, N)
    master_seed = _integer(master_seed, "master_seed", 0, 2)
    outer_id = _integer(outer_id, "outer_id", 0, MAX_OUTER - 1)
    purpose_id = _integer(purpose_id, "purpose_id", 1, 1)
    inner_id = _integer(inner_id, "inner_id", 0, MAX_INNER - 1)
    namespace = _integer(namespace, "namespace", 0, MAX_ID)
    if namespace not in (MAIN_NAMESPACE, SMOKE_NAMESPACE):
        raise ValueError("namespace must be the fixed M06 main or smoke namespace")
    sequence = np.random.SeedSequence(entropy=master_seed, spawn_key=(
        namespace, DISTRIBUTION_IDS[distribution], n, outer_id, purpose_id, inner_id))
    return np.random.Generator(np.random.PCG64(sequence))


def generate_bootstrap_means(sample, distribution, n=N, master_seed=0, outer_id=0,
                             inner_id=0, Bmax=MAX_B, namespace=MAIN_NAMESPACE,
                             max_elements=MAX_ELEMENTS, return_indices=False):
    """Draw one bounded stream with int64 RNG indices; optional uint8 audit copy.

    The saved audit indices are reversible because n=20. Only the caller's 18
    preregistered streams request them; no matrices from other streams persist.
    B prefixes and methods must reuse this returned mean vector, without a new
    call. Changing batching preserves the exact int64 RNG stream.
    """
    rng = rng_for_panel(distribution, n, master_seed, outer_id, 1, inner_id, namespace)
    raw = np.asarray(sample)
    if raw.ndim != 1 or raw.size != n or raw.dtype.kind not in "iuf":
        raise ValueError("sample must be a real vector of length20")
    with np.errstate(over="ignore", invalid="ignore"):
        sample = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(sample).all():
        raise ValueError("sample_nonfinite")
    if np.all(sample == sample[0]):
        raise ValueError("sample_constant")
    Bmax = _integer(Bmax, "Bmax", 1, MAX_B)
    max_elements = _integer(max_elements, "max_elements", N, MAX_ELEMENTS)
    if not isinstance(return_indices, (bool, np.bool_)):
        raise ValueError("return_indices must be boolean")
    batch = max_elements // N
    means = np.empty(Bmax, dtype=np.float64)
    saved_indices = np.empty((Bmax, N), dtype=np.uint8) if return_indices else None
    for start in range(0, Bmax, batch):
        count = min(batch, Bmax - start)
        indices = rng.integers(0, N, size=(count, N), dtype=np.int64)
        with np.errstate(over="ignore", invalid="ignore"):
            means[start:start + count] = np.mean(sample[indices], axis=1)
        if return_indices:
            saved_indices[start:start + count] = indices.astype(np.uint8)
    if not np.isfinite(means).all():
        raise ValueError("bootstrap_mean_nonfinite")
    return (means, saved_indices) if return_indices else means


def infer_prefixes(context, means, B_values=B_VALUES):
    """Unchanged inferential contract, with B-specific bias and quantiles."""
    return stability.infer_prefixes(context, means, B_values=B_values)


def original_metrics(data, se, n=N):
    """Reduce one original's (B,3 methods,K>=2), retaining local invalid counts.

    t uses effective K=1 and is checked invariant across K and B; its endpoint
    SDs are exactly0. Bootstrap SDs use ddof1 within the original. Invalid data
    affect only that method/B, never become noncoverage, and retain counts.
    The population SE in Apop is evaluation-only. Vse=Ase**2 is descriptive.
    """
    _integer(n, "n", N, N)
    try:
        lo, hi = np.asarray(data["lower"]), np.asarray(data["upper"])
        valid, warning = np.asarray(data["valid"]), np.asarray(data["tail_warning"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("data requires lower/upper/valid/tail_warning") from exc
    if lo.ndim != 3 or lo.shape[0] < 1 or lo.shape[1] != 3 or lo.shape[2] < 2:
        raise ValueError("endpoint shape must be (B,3 methods,K>=2)")
    if lo.dtype.kind not in "iuf" or hi.dtype.kind not in "iuf" or any(
            x.shape != lo.shape for x in (hi, valid, warning)) or valid.dtype.kind != "b" or warning.dtype.kind != "b":
        raise ValueError("endpoint real arrays and boolean masks must match")
    lo, hi = lo.astype(float), hi.astype(float)
    for array in (lo, hi, valid, warning):
        if not np.array_equal(array[:, 0], np.full_like(array[:, 0], array[0, 0, 0]), equal_nan=True):
            raise ValueError("t control must be invariant across inner streams and B")
    keys = ("lower_sd", "upper_sd", "S", "Ase", "Apop", "Vse", "C", "Mbelow", "Mabove", "T", "mean_width", "warning_rate")
    shape = lo.shape[:2]
    output = {key: np.full(shape, np.nan) for key in keys}
    output.update(complete=np.zeros(shape, bool), total_inner=np.zeros(shape, int),
                  valid_inner=np.zeros(shape, int), invalid_inner=np.zeros(shape, int))
    try:
        scale_valid = isinstance(se, Real) and not isinstance(se, (bool, np.bool_)) and np.isfinite(float(se)) and se > 0
    except (OverflowError, TypeError, ValueError):
        scale_valid = False
    for bi in range(shape[0]):
        for mi, method in enumerate(METHODS):
            selection = slice(0, 1) if method == "t" else slice(None)
            lower, upper = lo[bi, mi, selection], hi[bi, mi, selection]
            with np.errstate(over="ignore", invalid="ignore"):
                width = upper - lower
            mask = valid[bi, mi, selection] & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper) & np.isfinite(width)
            count, total = int(mask.sum()), mask.size
            output["valid_inner"][bi, mi], output["total_inner"][bi, mi] = count, total
            output["invalid_inner"][bi, mi] = total - count
            if not scale_valid or count != total:
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                sd_low = 0. if np.all(lower == lower[0]) else float(np.std(lower, ddof=1))
                sd_high = 0. if np.all(upper == upper[0]) else float(np.std(upper, ddof=1))
                S = np.hypot(sd_low, sd_high) / np.sqrt(2)
                ase = S / se
                values = {"lower_sd": sd_low, "upper_sd": sd_high, "S": S,
                          "Ase": ase, "Apop": S * np.sqrt(N), "Vse": ase**2,
                          "C": np.mean((lower <= 0) & (0 <= upper)),
                          "Mbelow": np.mean(upper < 0), "Mabove": np.mean(lower > 0),
                          "T": np.mean(upper < 0) - np.mean(lower > 0),
                          "mean_width": np.mean(width),
                          "warning_rate": np.mean(warning[bi, mi, selection])}
            if not np.isfinite(list(values.values())).all():
                continue
            output["complete"][bi, mi] = True
            for key, value in values.items():
                output[key][bi, mi] = value
    return output


def reduction_status(low, high, complete=True):
    """H2 per-population mean(A4999-.75*A999): strict upper<0 support."""
    base = {"status": "inconclusive", "reason": "incomplete_or_undefined_primary_comparison"}
    if not complete:
        return base
    try:
        finite = np.isfinite([low, high]).all()
    except (TypeError, ValueError):
        finite = False
    if not finite or low >= high:
        return base | {"reason": "undefined_primary_interval"}
    if high < 0:
        return {"status": "supported", "reason": ""}
    if low >= 0:
        return {"status": "not_supported", "reason": ""}
    return base | {"reason": "zero_touched_or_crossed"}


def combine_reduction_status(statuses):
    """Both populations must support H2; any contrary result has priority."""
    statuses = tuple(s["status"] if isinstance(s, dict) else s for s in statuses)
    if len(statuses) != 2 or any(s not in ("supported", "not_supported", "inconclusive") for s in statuses):
        raise ValueError("exactly two valid population statuses are required")
    if "not_supported" in statuses:
        return "not_supported"
    return "supported" if all(s == "supported" for s in statuses) else "inconclusive"
