"""RQ-M09 original-generation rules and saved-cohort validation.

Nothing is sampled at import, reference load, validation or receipt creation.
Only the guarded runner may call generate_cohort, after verifying preregistered
HEAD/source/environment, and must persist/read back all60 before any inner draw.
"""

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess

import numpy as np

from mathscape import allocation, matched_moments
from scripts import run_allocation_benchmark as accepted_runner

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_HEAD = "85382fd3ae8966dcf197d4006de21ea6edf9ef9f"
MAIN_NAMESPACE, SMOKE_NAMESPACE = 0x4D3039, 0x534D3039
DISTRIBUTIONS, DISTRIBUTION_IDS = allocation.DISTRIBUTIONS, allocation.DISTRIBUTION_IDS
METHODS, ALLOCATIONS = allocation.METHODS, allocation.ALLOCATIONS
N, ORIGINALS_PER_CELL, BUDGET_REPEATS, STREAMS, MAX_B = 20, 10, 20, 120, 4999
MAX_ELEMENTS = 1_000_000
REFERENCE_PATHS = tuple(f"evidence/rq_m08/{name}" for name in
    ("manifest.json", "hypotheses.json", "pooled_summary.csv")) + tuple(
        f"evidence/rq_m08/panels/{d}_seed{s}.npz" for d in DISTRIBUTIONS for s in (0,1,2))
NUMERIC_RTOL, NUMERIC_ATOL = allocation.NUMERIC_RTOL, allocation.NUMERIC_ATOL
IDENTITY_RTOL, IDENTITY_ATOL = allocation.IDENTITY_RTOL, allocation.IDENTITY_ATOL
prepare_sample, infer_single = allocation.prepare_sample, allocation.infer_single
original_metrics, budget_metrics = allocation.original_metrics, allocation.budget_metrics
mean_t_interval, band_status = allocation.mean_t_interval, allocation.band_status
variance_band_status, joint_status = allocation.variance_band_status, allocation.joint_status
sample_sha256 = allocation.sample_sha256


def _integer(value,name,low,high):
    return matched_moments._integer(value,name,low,high)


def _identity(distribution,n,master_seed,outer_id,namespace):
    distribution=matched_moments._distribution(distribution)
    n=_integer(n,"n",20,20);seed=_integer(master_seed,"master_seed",0,2)
    outer=_integer(outer_id,"outer_id",0,9);namespace=_integer(namespace,"namespace",0,2**32-1)
    if namespace not in (MAIN_NAMESPACE,SMOKE_NAMESPACE):raise ValueError("M09 namespace required")
    return distribution,n,seed,outer,namespace


def original_key(distribution,n=N,master_seed=0,outer_id=0,namespace=MAIN_NAMESPACE):
    """Six-element original key, never padded to the inner seven-element key."""
    dist,n,_,outer,namespace=_identity(distribution,n,master_seed,outer_id,namespace)
    return (namespace,DISTRIBUTION_IDS[dist],n,outer,0,0)


def stream_key(distribution,n=N,master_seed=0,outer_id=0,budget_repeat_id=0,
               stream_id=0,namespace=MAIN_NAMESPACE):
    dist,n,_,outer,namespace=_identity(distribution,n,master_seed,outer_id,namespace)
    repeat=_integer(budget_repeat_id,"budget_repeat_id",0,19)
    stream=_integer(stream_id,"stream_id",0,119)
    return (namespace,DISTRIBUTION_IDS[dist],n,outer,1,repeat,stream)


def rng_for_original(distribution,n=N,master_seed=0,outer_id=0,namespace=MAIN_NAMESPACE):
    key=original_key(distribution,n,master_seed,outer_id,namespace)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=int(master_seed),spawn_key=key)))


def rng_for_panel(distribution,n=N,master_seed=0,outer_id=0,purpose_id=1,
                  budget_repeat_id=0,stream_id=0,namespace=MAIN_NAMESPACE):
    _integer(purpose_id,"purpose_id",1,1)
    key=stream_key(distribution,n,master_seed,outer_id,budget_repeat_id,stream_id,namespace)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=int(master_seed),spawn_key=key)))


def generation_design():
    """Preregister rules and keys ONLY; no nonexistent new-cohort hashes."""
    namespaces={f"M{i:02d}_{kind}": (0x4D3030+i if kind=="main" else 0x534D3030+i)
                for i in range(2,10) for kind in ("main","smoke")}
    if len(set(namespaces.values()))!=len(namespaces):raise ValueError("namespace registry collision")
    originals=[dict(distribution=d,master_seed=s,outer_id=o,n=20,entropy=s,
                    spawn_key=list(original_key(d,20,s,o)))
               for d in DISTRIBUTIONS for s in (0,1,2) for o in range(10)]
    return dict(question_id="RQ-M09",generator="Generator(PCG64(SeedSequence(...)))",
        namespace_registry=namespaces,main_namespace=MAIN_NAMESPACE,smoke_namespace=SMOKE_NAMESPACE,
        distribution_ids=DISTRIBUTION_IDS.copy(),original_key_length=6,inner_key_length=7,
        originals=originals,budget_repeat_ids=list(range(20)),stream_ids=list(range(120)),
        inner_key_fields=["namespace","distribution_id","n","outer_id","purpose1","budget_repeat_id","stream_id"],
        original_count=60,inner_stream_count=144000,
        sample_hashes_known_before_generation=False,
        source_transform="accepted matched_moments Python-math constants and exact NumPy vector operation order; no sample standardization",
        constants={d:matched_moments.population_info(d)["constants"] for d in DISTRIBUTIONS},
        smoke_original_source="accepted M08 saved originals only, no new M09 originals",
        excluded_from_key=["J","K","method","execution_order","batch","attempt"])


def transform_raw(raw,distribution):
    """Pure accepted float64 operation order, suitable for saved-raw readback."""
    dist=matched_moments._distribution(distribution);raw=np.asarray(raw)
    if raw.shape!=(20,) or raw.dtype.kind not in "iuf":raise ValueError("raw must be a real length20 vector")
    raw=np.asarray(raw,dtype="<f8",order="C")
    if not np.isfinite(raw).all() or (dist=="gamma" and np.any(raw<0)):raise ValueError("invalid raw population values")
    with np.errstate(over="ignore",invalid="ignore"):
        sample=(raw-1.0 if dist=="gamma" else
            (np.exp(matched_moments.LOGNORMAL_S*raw)-matched_moments.LOGNORMAL_M)/matched_moments.LOGNORMAL_D)
    if not np.isfinite(sample).all():raise ValueError("original_sample_nonfinite")
    return np.asarray(sample,dtype="<f8",order="C")


def draw_original(distribution,n=N,master_seed=0,outer_id=0,namespace=MAIN_NAMESPACE,rng=None):
    """Draw exactly once in a verified main attempt; tests supply a fake rng.

    SMOKE deliberately cannot generate originals: its originals must come from
    the accepted M08 reference. Real main calls are forbidden before preregister.
    """
    dist,n,seed,outer,namespace=_identity(distribution,n,master_seed,outer_id,namespace)
    if namespace!=MAIN_NAMESPACE:raise ValueError("smoke must reuse accepted M08 saved originals")
    key=original_key(dist,n,seed,outer,namespace)
    rng=rng if rng is not None else rng_for_original(dist,n,seed,outer,namespace)
    raw=rng.gamma(shape=1.0,scale=1.0,size=n) if dist=="gamma" else rng.standard_normal(n)
    raw=np.asarray(raw,dtype="<f8",order="C");sample=transform_raw(raw,dist)
    return dict(distribution=dist,master_seed=seed,outer_id=outer,n=n,namespace=namespace,
        entropy=seed,spawn_key=list(key),sample=sample.copy(),raw=raw.copy(),
        raw_kind="gamma_y" if dist=="gamma" else "normal_z",
        constants=matched_moments.population_info(dist)["constants"],
        sample_sha256=sample_sha256(sample),raw_sha256=sample_sha256(raw))


def _verified_reference_blob(root,path):
    try:
        local=(root/path).read_bytes()
        blob=subprocess.run(["git","show",f"{ACCEPTED_HEAD}:{path}"],cwd=root,check=True,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
    except (OSError,subprocess.CalledProcessError) as exc:raise ValueError(f"accepted M08 reference unavailable: {path}") from exc
    if local!=blob:raise ValueError(f"M08 reference differs from accepted Git blob: {path}")
    return local


def _inspect_reference(root):
    root=Path(root).resolve();blobs={path:_verified_reference_blob(root,path) for path in REFERENCE_PATHS}
    try:
        manifest=json.loads(blobs["evidence/rq_m08/manifest.json"])
        hypotheses=json.loads(blobs["evidence/rq_m08/hypotheses.json"])
        pooled=list(csv.DictReader(StringIO(blobs["evidence/rq_m08/pooled_summary.csv"].decode())))
    except (ValueError,UnicodeError) as exc:raise ValueError("M08 reference JSON/CSV malformed") from exc
    if manifest.get("question_id")!="RQ-M08" or hypotheses.get("question_id")!="RQ-M08" or len(pooled)!=18:
        raise ValueError("M08 reference role/design mismatch")
    cells={}
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            path=f"evidence/rq_m08/panels/{dist}_seed{seed}.npz"
            try:
                with np.load(BytesIO(blobs[path]),allow_pickle=False) as panel:
                    for name,expected in dict(distribution=dist,master_seed=seed,n=20,O=10,R=20,
                            streams_per_budget=120,B=4999,alpha=.05,namespace=allocation.MAIN_NAMESPACE).items():
                        if panel[name].shape!=() or panel[name].item()!=expected:raise ValueError(f"M08 reference metadata mismatch: {name}")
                    samples,ids,hashes=panel["samples"],panel["outer_id"],panel["sample_sha256"]
                    if samples.shape!=(10,20) or samples.dtype!=np.dtype("<f8") or not np.isfinite(samples).all() or ids.dtype!=np.dtype("uint16") or not np.array_equal(ids,np.arange(10)) or hashes.shape!=(10,) or hashes.dtype!=np.dtype("S64"):
                        raise ValueError("M08 reference sample/ID/hash shape mismatch")
                    if not np.array_equal(hashes,np.array([sample_sha256(x).encode() for x in samples],dtype="S64")):
                        raise ValueError("M08 reference sample hash mismatch")
                    cell=dict(samples=samples.copy(),outer_id=ids.copy(),sample_sha256=hashes.copy())
            except (OSError,EOFError,KeyError) as exc:raise ValueError(f"M08 reference panel cannot be read: {path}") from exc
            for array in cell.values():array.flags.writeable=False
            cells[(dist,seed)]=cell
    reference=dict(schema_version=1,question_id="RQ-M09",reference_question_id="RQ-M08",
        accepted_head=ACCEPTED_HEAD,cohort_role="accepted_M08_reference_only",
        source_files_sha256={path:sha256(blob).hexdigest() for path,blob in blobs.items()},
        old_sample_count=60,selection_rule="all accepted M08 G/L x seed0,1,2 x outer0..9; smoke uses outer0 only",
        new_original_hashes_preregistered=False)
    return reference,dict(cohort_role="accepted_M08_reference_only",reference=reference,
        manifest=manifest,hypotheses=hypotheses,pooled=pooled,cells=cells)


def make_reference_manifest(root=ROOT):return _inspect_reference(root)[0]


def validate_reference_manifest(reference,root=ROOT):
    if not isinstance(reference,dict) or reference!=_inspect_reference(root)[0]:raise ValueError("M08 reference manifest mismatch")


def load_reference(reference=None,root=ROOT):
    expected,loaded=_inspect_reference(root)
    if reference is not None and reference!=expected:raise ValueError("M08 reference manifest mismatch")
    return loaded


def _pack_records(records):
    first=records[0];constants=first["constants"]
    result=dict(samples=np.asarray([r["sample"] for r in records],dtype="<f8"),
        raw=np.asarray([r["raw"] for r in records],dtype="<f8"),
        sample_sha256=np.array([r["sample_sha256"].encode() for r in records],dtype="S64"),
        raw_sha256=np.array([r["raw_sha256"].encode() for r in records],dtype="S64"),
        outer_id=np.array([r["outer_id"] for r in records],dtype=np.uint16),
        original_keys=np.array([r["spawn_key"] for r in records],dtype=np.uint32),
        distribution=np.array(first["distribution"]),master_seed=np.array(first["master_seed"]),
        n=np.array(20),namespace=np.array(first["namespace"]),raw_kind=np.array(first["raw_kind"]),
        constant_names=np.array(list(constants)),constant_values=np.array(list(constants.values()),dtype="<f8"))
    for array in result.values():array.flags.writeable=False
    return result


def _old_hashes(reference):
    if not isinstance(reference,dict) or reference.get("cohort_role")!="accepted_M08_reference_only" or "cells" not in reference:
        raise ValueError("verified M08 reference required for cohort isolation")
    return {sample_sha256(row) for cell in reference["cells"].values() for row in cell["samples"]}


def generate_cohort(reference,namespace=MAIN_NAMESPACE,draw_fn=None,progress=None):
    """Only after guarded prereg verification; generate60 once, no file writes.

    Progress retains each completed raw/sample record BEFORE collision checks,
    so unexpected old/duplicate matches can be preserved and stopped, not redrawn.
    Runner must save all six cells+receipt, then load_cohort before inner draws.
    """
    if namespace!=MAIN_NAMESPACE:raise ValueError("new cohort belongs to the main namespace only")
    old,seen=_old_hashes(reference),set();draw=draw_fn or draw_original
    progress=progress if progress is not None else {}
    cells={};records=[];progress.update(cells=cells,completed_records=records,generated_originals=0)
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            local=[]
            for outer in range(10):
                progress.update(active_distribution=dist,active_master_seed=seed,active_outer_id=outer)
                record=draw(dist,n=20,master_seed=seed,outer_id=outer,namespace=namespace)
                local.append(record);records.append(record);cells[(dist,seed)]=_pack_records(local)
                progress["generated_originals"]+=1
                digest=sample_sha256(record["sample"])
                if digest in old:raise ValueError("unexpected exact M08 sample match; no replacement")
                if digest in seen:raise ValueError("unexpected duplicate new-cohort sample; no replacement")
                seen.add(digest)
    validate_cohort(cells,reference,namespace)
    return cells


def validate_cohort(cells,reference,namespace=MAIN_NAMESPACE):
    """Read-only raw->x/key/bytes/metadata/hash check; never draws a random value."""
    if namespace!=MAIN_NAMESPACE:raise ValueError("new cohort main namespace required")
    expected={(d,s) for d in DISTRIBUTIONS for s in (0,1,2)}
    if set(cells)!=expected:raise ValueError("new cohort must contain all six fixed cells")
    old,seen=_old_hashes(reference),set()
    for dist,seed in sorted(expected):
        cell=cells[(dist,seed)]
        required={"samples","raw","sample_sha256","raw_sha256","outer_id","original_keys",
                  "distribution","master_seed","n","namespace","raw_kind","constant_names","constant_values"}
        if not isinstance(cell,dict) or not required<=cell.keys():
            raise ValueError("new cohort raw/key metadata required; M08 reference cells are not main originals")
        def scalar(name):
            value=np.asarray(cell[name])
            if value.shape!=():raise ValueError(f"new cohort metadata must be scalar: {name}")
            return value.item()
        for name,value in dict(distribution=dist,master_seed=seed,n=20,namespace=namespace,
                               raw_kind="gamma_y" if dist=="gamma" else "normal_z").items():
            if scalar(name)!=value:raise ValueError(f"new cohort metadata mismatch: {name}")
        constants=matched_moments.population_info(dist)["constants"]
        if list(np.asarray(cell["constant_names"]).astype(str))!=list(constants) or np.asarray(cell["constant_values"]).dtype!=np.dtype("<f8") or np.asarray(cell["constant_values"]).tobytes()!=np.asarray(list(constants.values()),dtype="<f8").tobytes():
            raise ValueError("new cohort analytic constants differ")
        sample,raw=np.asarray(cell["samples"]),np.asarray(cell["raw"])
        ids,keys=np.asarray(cell["outer_id"]),np.asarray(cell["original_keys"])
        if sample.shape!=(10,20) or raw.shape!=(10,20) or sample.dtype!=np.dtype("<f8") or raw.dtype!=np.dtype("<f8") or not np.isfinite(sample).all() or not np.isfinite(raw).all():raise ValueError("new cohort raw/sample shape/dtype/nonfinite mismatch")
        if ids.dtype!=np.dtype("uint16") or not np.array_equal(ids,np.arange(10)) or keys.shape!=(10,6) or keys.dtype!=np.dtype("uint32"):raise ValueError("new cohort ID/key shape/dtype mismatch")
        for name in ("sample_sha256","raw_sha256"):
            if np.asarray(cell[name]).shape!=(10,) or np.asarray(cell[name]).dtype!=np.dtype("S64"):raise ValueError("new cohort hash shape/dtype mismatch")
        for outer in range(10):
            if tuple(keys[outer])!=original_key(dist,20,seed,outer,namespace):raise ValueError("new original six-element key mismatch")
            transformed=transform_raw(raw[outer],dist)
            if transformed.tobytes()!=sample[outer].tobytes():raise ValueError("saved raw transform/sample bytes mismatch")
            digest=sample_sha256(sample[outer]);raw_digest=sample_sha256(raw[outer])
            if digest.encode()!=cell["sample_sha256"][outer] or raw_digest.encode()!=cell["raw_sha256"][outer]:raise ValueError("new cohort saved hash mismatch")
            if digest in old:raise ValueError("unexpected exact M08 sample match; no replacement")
            if digest in seen:raise ValueError("unexpected duplicate new-cohort sample; no replacement")
            seen.add(digest)
    return dict(original_samples=60,unique_new_sample_hashes=len(seen),exact_M08_matches=0,
        duplicate_new_samples=0,raw_transform_bytes_verified=True,
        interpretation="key separation and no exact sample matches are implementation checks, not a proof of statistical independence")


def cohort_receipt(cells,reference,source_commit,namespace=MAIN_NAMESPACE):
    if not isinstance(source_commit,str) or len(source_commit)!=40 or any(c not in "0123456789abcdef" for c in source_commit):raise ValueError("full source commit required")
    summary=validate_cohort(cells,reference,namespace);records=[]
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            cell=cells[(dist,seed)]
            constants=dict(zip(cell["constant_names"].astype(str),[float(x) for x in cell["constant_values"]]))
            for outer in range(10):
                records.append(dict(distribution=dist,master_seed=seed,outer_id=outer,n=20,namespace=namespace,
                    entropy=seed,spawn_key=[int(x) for x in cell["original_keys"][outer]],raw_kind=cell["raw_kind"].item(),
                    constants=constants,sample_sha256=cell["sample_sha256"][outer].decode(),raw_sha256=cell["raw_sha256"][outer].decode()))
    return dict(schema_version=1,question_id="RQ-M09",source_git_sha=source_commit,cohort_role="new_M09_main_originals",
        namespace=namespace,hash_recording_phase="after generation inside the verified main attempt, never preregistered before creation",
        original_samples=60,records=records,isolation_checks=summary,
        M08_reference_accepted_head=ACCEPTED_HEAD,M08_reference_files_sha256=reference.get("reference",{}).get("source_files_sha256",{}))


def load_cohort(directory,reference,namespace=MAIN_NAMESPACE):
    cells={};directory=Path(directory)
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            with np.load(directory/f"{dist}_seed{seed}.npz",allow_pickle=False) as panel:
                cell={name:panel[name] for name in panel.files}
            for array in cell.values():array.flags.writeable=False
            cells[(dist,seed)]=cell
    validate_cohort(cells,reference,namespace)
    return cells


def generate_bootstrap_means(sample,distribution,n=N,master_seed=0,outer_id=0,
        budget_repeat_id=0,stream_id=0,Bmax=MAX_B,namespace=MAIN_NAMESPACE,
        max_elements=MAX_ELEMENTS,return_indices=False):
    rng=rng_for_panel(distribution,n,master_seed,outer_id,1,budget_repeat_id,stream_id,namespace)
    raw=np.asarray(sample)
    if raw.ndim!=1 or raw.size!=n or raw.dtype.kind not in "iuf":raise ValueError("sample must be real length20 vector")
    sample=raw.astype(float)
    if not np.isfinite(sample).all() or np.all(sample==sample[0]):raise ValueError("sample is nonfinite or constant")
    Bmax=_integer(Bmax,"Bmax",1,MAX_B);cap=_integer(max_elements,"max_elements",20,MAX_ELEMENTS)
    if not isinstance(return_indices,(bool,np.bool_)):raise ValueError("return_indices must be boolean")
    means=np.empty(Bmax);saved=np.empty((Bmax,20),np.uint8) if return_indices else None
    for start in range(0,Bmax,cap//20):
        count=min(cap//20,Bmax-start);indices=rng.integers(0,20,size=(count,20),dtype=np.int64)
        with np.errstate(over="ignore",invalid="ignore"):means[start:start+count]=sample[indices].mean(axis=1)
        if return_indices:saved[start:start+count]=indices.astype(np.uint8)
    if not np.isfinite(means).all():raise ValueError("bootstrap_mean_nonfinite")
    return (means,saved) if return_indices else means


def summarize(rows,pooled=False):return accepted_runner.summarize(rows,pooled=pooled)


def assemble_hypotheses(comparisons):
    """Only the question ID changes; M08 outcomes are never an input."""
    return {**accepted_runner.assemble_hypotheses(comparisons),"question_id":"RQ-M09"}


def decomposition_summary(comparisons,decompositions):
    return accepted_runner.decomposition_summary(comparisons,decompositions)


def cohort_comparison(new_hypotheses,reference):
    """Descriptive side-by-side only: no pooling, paired IDs or new test."""
    if new_hypotheses.get("question_id")!="RQ-M09" or reference["hypotheses"].get("question_id")!="RQ-M08":raise ValueError("separate cohort question IDs required")
    return dict(comparison_role="descriptive only, no pooling or inter-cohort hypothesis test",
        cohorts=[dict(question_id="RQ-M08",accepted_head=ACCEPTED_HEAD,hypotheses=reference["hypotheses"]),
                 dict(question_id="RQ-M09",hypotheses=new_hypotheses)])
