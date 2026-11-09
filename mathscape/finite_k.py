"""RQ-M07 finite-K measurement, paired decomposition and verified saved input.

K120 is a finite measurement, not an ideal endpoint or an absolute-bias
reference. This module generates bootstrap indices only, never originals.
"""

from hashlib import sha256
from io import BytesIO
import json
from numbers import Integral, Real
from pathlib import Path
import subprocess

import numpy as np

from mathscape import stability

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_HEAD = "e2feae23397d014f6aa3363773da9913da85fcf7"
M05_ACCEPTED_HEAD = "dc2501c3099f5c25227d9eee00d227c2f0b02f81"
M06_MANIFEST_PATH = "evidence/rq_m06/manifest.json"
INPUT_PATHS = (M06_MANIFEST_PATH,) + tuple(
    f"evidence/rq_m06/panels/{d}_seed{s}.npz"
    for d in ("gamma", "lognormal") for s in (0, 1, 2))
DISTRIBUTIONS = ("gamma", "lognormal")
DISTRIBUTION_IDS = {"gamma": 1, "lognormal": 2}
METHODS = ("t", "percentile", "bca")
K_VALUES = (30, 120)
N, J, GROUP_SIZE = 20, 20, 30
MAX_OUTER, MAX_INNER, MAX_B = 10, 120, 4999
MAX_ELEMENTS, MAX_ID = 1_000_000, 2**32 - 1
MAIN_NAMESPACE, SMOKE_NAMESPACE = 0x4D3037, 0x534D3037
# Locked against deterministic fixtures before the sole main experiment.
NUMERIC_RTOL, NUMERIC_ATOL = 2e-10, 2e-12
IDENTITY_RTOL, IDENTITY_ATOL = 2e-12, 2e-12
K_METRICS = ("lower_sd", "upper_sd", "S", "Ase", "Apop", "Vse", "C",
             "Mbelow", "Mabove", "T", "mean_width", "warning_rate")
GROUP_FIELDS = ("group_lower_mean", "group_upper_mean", "group_lower_var",
                "group_upper_var", "group_V", "group_A")
REPLICATE_FIELDS = ("V30", "A30", "V120", "A120", "Vbar", "Abar", "Jgap",
    "Pgap", "Rgap", "delta", "decomposition_residual", "decomposition_normalized_residual",
    "decomposition_abs_normalized_residual", "lower_ss_residual", "upper_ss_residual",
    "lower_ss_scale", "upper_ss_scale", "lower_ss_normalized_residual",
    "upper_ss_normalized_residual", "lower_ss_abs_normalized_residual",
    "upper_ss_abs_normalized_residual")
prepare_sample = stability.prepare_sample
mean_t_interval = stability.mean_t_interval


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(value)


def _distribution(distribution):
    if not isinstance(distribution, str) or distribution not in DISTRIBUTION_IDS:
        raise ValueError("distribution must be gamma or lognormal")
    return distribution


def sample_sha256(sample):
    return sha256(np.asarray(sample, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def _verified_blob(root, path):
    try:
        local = (root / path).read_bytes()
        accepted = subprocess.run(["git", "show", f"{ACCEPTED_HEAD}:{path}"], cwd=root,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"accepted saved input unavailable: {path}") from exc
    if local != accepted:
        raise ValueError(f"saved input differs from accepted Git blob: {path}")
    return local


def _scalar(panel, key, expected):
    if key not in panel or panel[key].shape != () or panel[key].item() != expected:
        raise ValueError(f"saved input metadata mismatch: {key}")


def _inspect_inputs(root):
    root = Path(root).resolve()
    blobs = {path: _verified_blob(root, path) for path in INPUT_PATHS}
    try:
        source = json.loads(blobs[M06_MANIFEST_PATH])
    except (ValueError, UnicodeError) as exc:
        raise ValueError("M06 manifest JSON is invalid") from exc
    expected = dict(distributions=list(DISTRIBUTIONS), n=20, master_seeds=[0, 1, 2],
                    R=200, K=30, B_values=[999, 4999], alpha=.05, methods=list(METHODS))
    if source.get("question_id") != "RQ-M06" or any(
            source.get("design", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("M06 manifest design mismatch")
    upstream = source.get("inputs", {})
    if upstream.get("accepted_head") != M05_ACCEPTED_HEAD:
        raise ValueError("M05 accepted input provenance mismatch")
    paths05 = ("evidence/rq_m05/manifest.json",) + tuple(
        f"evidence/rq_m05/panels/{d}_seed{s}.npz" for d in DISTRIBUTIONS for s in (0, 1, 2))
    upstream_hashes = upstream.get("source_files_sha256", {})
    if set(upstream_hashes) != set(paths05):
        raise ValueError("M05 upstream file set mismatch")
    old_blobs = {path: _verified_blob(root, path) for path in paths05}
    if any(sha256(blob).hexdigest() != upstream_hashes[path] for path, blob in old_blobs.items()):
        raise ValueError("M05 upstream saved input hash mismatch")
    try:
        provenance = {(s["distribution"], s["master_seed"]): s for s in upstream["selections"]}
    except (KeyError, TypeError) as exc:
        raise ValueError("M06 selected input provenance is malformed") from exc
    if len(provenance) != 6 or len(upstream["selections"]) != 6:
        raise ValueError("M06 upstream selection cell set mismatch")
    cells, selections = {}, []
    for dist in DISTRIBUTIONS:
        for seed in (0, 1, 2):
            path = f"evidence/rq_m06/panels/{dist}_seed{seed}.npz"
            old_path = f"evidence/rq_m05/panels/{dist}_seed{seed}.npz"
            try:
                with np.load(BytesIO(blobs[path]), allow_pickle=False) as panel, np.load(
                        BytesIO(old_blobs[old_path]), allow_pickle=False) as old:
                    for key, value in dict(distribution=dist, master_seed=seed, n=20, R=200,
                            observed_original_count=200, K=30, alpha=.05, namespace=0x4D3036,
                            input_accepted_head=M05_ACCEPTED_HEAD).items():
                        _scalar(panel, key, value)
                    for key, value in dict(distribution=dist, master_seed=seed, n=20,
                            R=10000, K=1, B=4999, alpha=.05, namespace=0x4D3035).items():
                        _scalar(old, key, value)
                    if not np.array_equal(panel["B_values"], [999, 4999]) or not np.array_equal(panel["methods"], METHODS):
                        raise ValueError("M06 B/method metadata mismatch")
                    samples, ids, hashes = panel["samples"], panel["outer_id"], panel["sample_sha256"]
                    if samples.shape != (200, 20) or samples.dtype != np.dtype("<f8") or not np.isfinite(samples).all():
                        raise ValueError("M06 sample shape/dtype/nonfinite mismatch")
                    if ids.dtype != np.dtype("uint16") or not np.array_equal(ids, np.arange(200)):
                        raise ValueError("M06 original IDs mismatch")
                    calculated = np.array([sample_sha256(row).encode() for row in samples], dtype="S64")
                    if hashes.shape != (200,) or hashes.dtype != np.dtype("S64") or not np.array_equal(hashes, calculated):
                        raise ValueError("M06 original sample hash mismatch")
                    origin = provenance[(dist, seed)]
                    if origin["outer_ids"] != list(range(200)) or origin["sample_sha256"] != [h.decode() for h in hashes] or origin["selected_samples_sha256"] != sample_sha256(samples):
                        raise ValueError("M06 input manifest original correspondence mismatch")
                    old_samples, old_ids, old_hashes = old["samples"], old["outer_id"], old["sample_sha256"]
                    if old_samples.shape != (10000, 20) or not np.array_equal(old_ids[:200], ids) or not np.array_equal(old_hashes[:200], hashes) or old_samples[:200].astype("<f8").tobytes() != samples.astype("<f8").tobytes():
                        raise ValueError("M05 original bytes/hash/ID correspondence mismatch")
                    selected = dict(samples=samples[:10].copy(), outer_id=ids[:10].copy(),
                                    sample_sha256=hashes[:10].copy())
            except (OSError, KeyError, EOFError) as exc:
                raise ValueError(f"saved original input cannot be read: {path}") from exc
            for array in selected.values():
                array.flags.writeable = False
            cells[(dist, seed)] = selected
            selections.append(dict(distribution=dist, master_seed=seed, source_path=path,
                m05_source_path=old_path, outer_ids=list(range(10)),
                sample_sha256=[h.decode() for h in selected["sample_sha256"]],
                selected_samples_sha256=sample_sha256(selected["samples"])))
    inputs = dict(schema_version=1, question_id="RQ-M07", accepted_head=ACCEPTED_HEAD,
        source_question_id="RQ-M06", source_files_sha256={p: sha256(b).hexdigest() for p,b in blobs.items()},
        upstream_accepted_head=M05_ACCEPTED_HEAD, upstream_files_sha256=upstream_hashes,
        selection_rule="gamma/lognormal x seed0,1,2 exact M06 saved outer IDs0..9; never use endpoint outcomes",
        n=20, originals_per_cell=10, selected_originals=60,
        sample_hash_contract="SHA256 little-endian float64 C-order bytes", selections=selections)
    return inputs, cells


def make_inputs_manifest(root=ROOT):
    """Verify accepted M06/M05 saved bytes, then return deterministic provenance."""
    return _inspect_inputs(root)[0]


def validate_inputs_manifest(inputs, root=ROOT):
    if not isinstance(inputs, dict) or inputs != _inspect_inputs(root)[0]:
        raise ValueError("M07 input manifest differs from verified saved originals")


def load_inputs(inputs=None, root=ROOT):
    """Six read-only cells, samples(10,20)/outer_id/hash; no original generator."""
    expected, cells = _inspect_inputs(root)
    if inputs is not None and (not isinstance(inputs, dict) or inputs != expected):
        raise ValueError("M07 input manifest differs from verified saved originals")
    return cells


def rng_for_panel(distribution, n=N, master_seed=0, outer_id=0, purpose_id=1,
                  replicate_id=0, inner_id=0, namespace=MAIN_NAMESPACE):
    dist = _distribution(distribution)
    n = _integer(n, "n", 20, 20)
    seed = _integer(master_seed, "master_seed", 0, 2)
    outer = _integer(outer_id, "outer_id", 0, 9)
    purpose = _integer(purpose_id, "purpose_id", 1, 1)
    replicate = _integer(replicate_id, "replicate_id", 0, 19)
    inner = _integer(inner_id, "inner_id", 0, 119)
    namespace = _integer(namespace, "namespace", 0, MAX_ID)
    if namespace not in (MAIN_NAMESPACE, SMOKE_NAMESPACE):
        raise ValueError("namespace must be M07 main or smoke")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=seed,
        spawn_key=(namespace, DISTRIBUTION_IDS[dist], n, outer, purpose, replicate, inner))))


def generate_bootstrap_means(sample, distribution, n=N, master_seed=0, outer_id=0,
        replicate_id=0, inner_id=0, Bmax=MAX_B, namespace=MAIN_NAMESPACE,
        max_elements=MAX_ELEMENTS, return_indices=False):
    rng = rng_for_panel(distribution, n, master_seed, outer_id, 1, replicate_id, inner_id, namespace)
    raw = np.asarray(sample)
    if raw.ndim != 1 or raw.size != n or raw.dtype.kind not in "iuf":
        raise ValueError("sample must be a real length20 vector")
    sample = raw.astype(np.float64)
    if not np.isfinite(sample).all() or np.all(sample == sample[0]):
        raise ValueError("sample is nonfinite or constant")
    Bmax = _integer(Bmax, "Bmax", 1, MAX_B)
    limit = _integer(max_elements, "max_elements", N, MAX_ELEMENTS)
    if not isinstance(return_indices, (bool, np.bool_)):
        raise ValueError("return_indices must be boolean")
    means = np.empty(Bmax)
    saved = np.empty((Bmax, N), np.uint8) if return_indices else None
    batch = limit // N
    for start in range(0, Bmax, batch):
        count = min(batch, Bmax-start)
        indices = rng.integers(0, N, size=(count, N), dtype=np.int64)
        with np.errstate(over="ignore", invalid="ignore"):
            means[start:start+count] = np.mean(sample[indices], axis=1)
        if return_indices:
            saved[start:start+count] = indices.astype(np.uint8)
    if not np.isfinite(means).all():
        raise ValueError("bootstrap_mean_nonfinite")
    return (means, saved) if return_indices else means


def infer_single(context, means):
    """Both bootstrap methods use one shared vector; no truth inputs."""
    try:
        raw = np.asarray(means)
        size = raw.size if raw.ndim == 1 and 0 < raw.size <= MAX_B else 1
    except (TypeError, ValueError):
        size = 1
    return stability.infer_prefixes(context, means, (size,))[size]


def _variance(values):
    # np.var of repeated .8 may be positive from rounding its mean.
    return 0.0 if np.all(values == values[0]) else float(np.var(values, ddof=1))


def _valid_scale(se):
    try:
        return isinstance(se, Real) and not isinstance(se, (bool, np.bool_)) and np.isfinite(float(se)) and se > 0
    except (OverflowError, TypeError, ValueError):
        return False


def _endpoint_arrays(data, dimensions):
    try:
        lo, hi, valid, warning = [np.asarray(data[k]) for k in ("lower", "upper", "valid", "tail_warning")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("endpoint data requires lower/upper/valid/tail_warning") from exc
    if lo.ndim != dimensions or lo.shape[-2] != 3 or any(x.shape != lo.shape for x in (hi, valid, warning)) or lo.dtype.kind not in "iuf" or hi.dtype.kind not in "iuf" or valid.dtype.kind != "b" or warning.dtype.kind != "b":
        raise ValueError("endpoint arrays must have matching (...,3,K) real/boolean shapes")
    return lo.astype(float), hi.astype(float), valid, warning


def _t_invariant(arrays):
    lower,upper,valid,_ = arrays
    lo,hi,mask = lower[...,0,:],upper[...,0,:],valid[...,0,:]
    observed = mask & np.isfinite(lo) & np.isfinite(hi) & (lo<=hi)
    # Missing t slots invalidate t locally; their values/masks are not a
    # structural error and must never poison complete BCa observations.
    for control in (lo[observed],hi[observed]):
        if control.size and not np.all(control==control[0]):
            raise ValueError("valid observed t endpoints must be invariant across replicate/inner IDs")


def replicate_metrics(data, se, group_size=GROUP_SIZE):
    """One r's method-by-K endpoints, split into four nonoverlapping groups.

    Entire-method completeness requires every K endpoint, including ones beyond
    the first group. Invalid primary quantities remain NA; raw residuals from an
    algebra-check failure are retained with reason and algebra_ok=False.
    """
    g = _integer(group_size, "group_size", 2, GROUP_SIZE)
    lo, hi, valid, warning = arrays = _endpoint_arrays(data, 2)
    if lo.shape != (3, 4*g):
        raise ValueError("one replicate requires exactly four endpoint groups")
    _t_invariant(arrays)
    out = {key: np.full(3, np.nan) for key in REPLICATE_FIELDS}
    out.update({key: np.full((3, 4), np.nan) for key in GROUP_FIELDS})
    out.update(complete=np.zeros(3, bool), algebra_ok=np.zeros(3, bool),
        total_inner=np.zeros(3, int), valid_inner=np.zeros(3, int), invalid_inner=np.zeros(3, int),
        reason=np.full(3, "", dtype="U80"), by_K={key: np.full((2, 3), np.nan) for key in K_METRICS})
    for mi, method in enumerate(METHODS):
        lower, upper = lo[mi], hi[mi]
        with np.errstate(over="ignore", invalid="ignore"):
            width = upper-lower
        mask = valid[mi] & np.isfinite(lower) & np.isfinite(upper) & np.isfinite(width) & (lower <= upper)
        selection = slice(0, 1) if method == "t" else slice(None)
        total = mask[selection].size
        count = int(mask.all()) if method == "t" else int(mask.sum())
        out["total_inner"][mi], out["valid_inner"][mi], out["invalid_inner"][mi] = total, count, total-count
        if not _valid_scale(se) or not mask.all():
            out["reason"][mi] = "sample_se_invalid" if not _valid_scale(se) else "endpoint_invalid_or_nonfinite"
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            l_groups, u_groups = lower.reshape(4,g), upper.reshape(4,g)
            lmeans, umeans = l_groups.mean(axis=1), u_groups.mean(axis=1)
            lv, uv = np.array([_variance(x) for x in l_groups]), np.array([_variance(x) for x in u_groups])
            lv120, uv120 = _variance(lower), _variance(upper)
            denominator = 2*np.float64(se)*np.float64(se)
            group_V = (lv+uv)/denominator
            group_A = np.sqrt(group_V)
            V120 = (lv120+uv120)/denominator
            Vbar, Abar, A120 = float(group_V.mean()), float(group_A.mean()), np.sqrt(V120)
            Jgap, Pgap, Rgap = np.sqrt(Vbar)-Abar, A120-np.sqrt(Vbar), Abar-group_A[0]
            delta = A120-group_A[0]
            residual = delta-(Jgap+Pgap+Rgap)
            ss = []
            for var120, group_var, group_mean, endpoint in ((lv120,lv,lmeans,lower),(uv120,uv,umeans,upper)):
                lhs = (4*g-1)*var120
                rhs = (g-1)*group_var.sum()+g*np.sum((group_mean-endpoint.mean())**2)
                ss.append((lhs-rhs,max(abs(lhs),abs(rhs))))
            values = dict(V30=group_V[0], A30=group_A[0], V120=V120, A120=A120,
                Vbar=Vbar,Abar=Abar,Jgap=Jgap,Pgap=Pgap,Rgap=Rgap,delta=delta,
                decomposition_residual=residual,decomposition_normalized_residual=residual/max(1.,abs(delta),abs(Jgap+Pgap+Rgap)),
                lower_ss_residual=ss[0][0],upper_ss_residual=ss[1][0],lower_ss_scale=ss[0][1],upper_ss_scale=ss[1][1],
                lower_ss_normalized_residual=ss[0][0]/max(1.,ss[0][1]),upper_ss_normalized_residual=ss[1][0]/max(1.,ss[1][1]))
            for key in ("decomposition", "lower_ss", "upper_ss"):
                values[key+"_abs_normalized_residual"] = abs(values[key+"_normalized_residual"])
            groups = dict(group_lower_mean=lmeans,group_upper_mean=umeans,
                          group_lower_var=lv,group_upper_var=uv,group_V=group_V,group_A=group_A)
        if method == "t":
            values = {key: 0. for key in REPLICATE_FIELDS}
            for key in ("group_lower_var", "group_upper_var", "group_V", "group_A"):
                groups[key] = np.zeros(4)
        if not np.isfinite(list(values.values())).all() or not all(np.isfinite(x).all() for x in groups.values()):
            out["reason"][mi] = "derived_statistic_nonfinite"
            continue
        for key, value in values.items(): out[key][mi] = value
        for key, value in groups.items(): out[key][mi] = value
        ss_ok = all(abs(residual) <= IDENTITY_ATOL+IDENTITY_RTOL*scale for residual,scale in ss)
        gap_ok = Jgap >= -(IDENTITY_ATOL+IDENTITY_RTOL*max(abs(np.sqrt(Vbar)),abs(Abar)))
        identity_ok = abs(values["decomposition_residual"]) <= IDENTITY_ATOL+IDENTITY_RTOL*max(abs(delta),abs(Jgap+Pgap+Rgap))
        out["algebra_ok"][mi] = ss_ok and gap_ok and identity_ok
        if not out["algebra_ok"][mi]:
            out["reason"][mi] = "algebraic_identity_failed"
            continue
        by_k = {}
        for ki, limit in enumerate((g, 4*g)):
            l, u, w = lower[:limit], upper[:limit], warning[mi,:limit]
            ls, us = (0.,0.) if method == "t" else (np.sqrt(_variance(l)),np.sqrt(_variance(u)))
            S = np.hypot(ls,us)/np.sqrt(2)
            A = 0. if method == "t" else (values["A30"] if ki == 0 else values["A120"])
            by_k[ki] = dict(lower_sd=ls,upper_sd=us,S=S,Ase=A,Apop=S*np.sqrt(20),Vse=A*A,
                C=np.mean((l<=0)&(0<=u)),Mbelow=np.mean(u<0),Mabove=np.mean(l>0),T=np.mean(u<0)-np.mean(l>0),
                mean_width=np.mean(u-l),warning_rate=np.mean(w))
        if not all(np.isfinite(list(v.values())).all() for v in by_k.values()):
            out["reason"][mi] = "derived_statistic_nonfinite"
            continue
        out["complete"][mi] = True
        for ki, values_k in by_k.items():
            for key,value in values_k.items():out["by_K"][key][ki,mi] = value
    return out


def original_metrics(data, se, group_size=GROUP_SIZE, expected_replicates=J):
    """Reduce all required r for one original; never pool r/K as originals."""
    expected = _integer(expected_replicates, "expected_replicates", 2, J)
    lo,hi,valid,warning = arrays = _endpoint_arrays(data,3)
    if lo.shape[0] != expected:
        raise ValueError("all planned replicate slots must be present, including invalid/missing slots")
    _t_invariant(arrays)
    results = [replicate_metrics(dict(lower=lo[r],upper=hi[r],valid=valid[r],tail_warning=warning[r]),se,group_size) for r in range(expected)]
    rep = {key:np.stack([row[key] for row in results]) for key in results[0] if key != "by_K"}
    rep["by_K"] = {key:np.stack([row["by_K"][key] for row in results]) for key in K_METRICS}
    float_fields = ("d","Q30","Q120","W","ratio_Q120_Q30","mean_A30","mean_A120","mean_V30","mean_V120",
        "mean_Jgap","mean_Pgap","mean_Rgap","mean_delta","max_abs_decomposition_residual",
        "max_abs_lower_ss_residual","max_abs_upper_ss_residual","max_abs_decomposition_normalized_residual",
        "max_abs_lower_ss_normalized_residual","max_abs_upper_ss_normalized_residual")
    original = {key:np.full(3,np.nan) for key in float_fields}
    original.update(complete=rep["complete"].all(axis=0),complete_replicates=rep["complete"].sum(axis=0),
        planned_replicates=np.full(3,expected,int),total_inner=rep["total_inner"].sum(axis=0),
        valid_inner=rep["valid_inner"].sum(axis=0),invalid_inner=rep["invalid_inner"].sum(axis=0))
    # One original's deterministic t interval, not J independent controls.
    original["total_inner"][0]=1
    original["valid_inner"][0]=int(np.all(rep["valid_inner"][:,0]==1))
    original["invalid_inner"][0]=1-original["valid_inner"][0]
    by_k = {key:np.full((2,3),np.nan) for key in K_METRICS}
    for mi in range(3):
        if not original["complete"][mi]:continue
        a30,a120=rep["A30"][:,mi],rep["A120"][:,mi]
        Q30,Q120 = (0.,0.) if mi==0 else (_variance(a30),_variance(a120))
        values=dict(d=float(rep["delta"][:,mi].mean()),Q30=Q30,Q120=Q120,W=Q120-.5*Q30,
                    mean_A30=float(a30.mean()),mean_A120=float(a120.mean()),
                    mean_V30=float(rep["V30"][:,mi].mean()),mean_V120=float(rep["V120"][:,mi].mean()))
        for key in ("Jgap","Pgap","Rgap","delta"):values["mean_"+key]=float(rep[key][:,mi].mean())
        for key in ("decomposition_residual","lower_ss_residual","upper_ss_residual",
                    "decomposition_normalized_residual","lower_ss_normalized_residual","upper_ss_normalized_residual"):
            values["max_abs_"+key]=float(np.max(np.abs(rep[key][:,mi])))
        with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
            local_k={key:rep["by_K"][key][:,:,mi].mean(axis=0) for key in K_METRICS}
            ratio=Q120/Q30 if Q30>0 else np.nan
        if not np.isfinite(list(values.values())).all() or not all(np.isfinite(v).all() for v in local_k.values()):
            original["complete"][mi]=False
            continue
        for key,value in values.items():original[key][mi]=value
        original["ratio_Q120_Q30"][mi]=ratio if np.isfinite(ratio) else np.nan
        for key,value in local_k.items():by_k[key][:,mi]=value
    return dict(replicate=rep,original=original,by_K=by_k)


def band_status(low, high, complete=True, margin=.0025):
    """H1 supports a small mean movement INSIDE the band (opposite M06 H1)."""
    if not _valid_scale(margin):raise ValueError("margin must be finite and positive")
    base=dict(status="inconclusive",reason="incomplete_or_undefined_primary_comparison")
    try:finite=np.isfinite([low,high]).all()
    except (TypeError,ValueError):finite=False
    if not complete or not finite or low>=high:return base
    if low>-margin and high<margin:return dict(status="supported",reason="")
    if low>margin or high<-margin:return dict(status="not_supported",reason="")
    return dict(status="inconclusive",reason="margin_touched_or_crossed")


def reduction_status(low, high, complete=True):
    """H2 concerns repeated-measurement VARIANCE, not endpoint SD reduction."""
    base=dict(status="inconclusive",reason="incomplete_or_undefined_primary_comparison")
    try:finite=np.isfinite([low,high]).all()
    except (TypeError,ValueError):finite=False
    if not complete or not finite or low>=high:return base
    if high<0:return dict(status="supported",reason="")
    if low>=0:return dict(status="not_supported",reason="")
    return dict(status="inconclusive",reason="zero_touched_or_crossed")


def joint_status(statuses):
    values=tuple(s["status"] if isinstance(s,dict) else s for s in statuses)
    if len(values)!=2 or any(s not in ("supported","not_supported","inconclusive") for s in values):
        raise ValueError("exactly two valid population statuses required")
    if "not_supported" in values:return "not_supported"
    return "supported" if all(s=="supported" for s in values) else "inconclusive"
