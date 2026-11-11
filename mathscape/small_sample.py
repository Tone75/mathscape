"""RQ-M10 original-generation rules and saved-cohort validation.

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
ACCEPTED_HEAD = "1bf925b55a838650ee020fe09f806cf64e8ebc02"
MAIN_NAMESPACE, SMOKE_NAMESPACE = 0x4D3130, 0x534D3130
DISTRIBUTIONS, DISTRIBUTION_IDS = allocation.DISTRIBUTIONS, allocation.DISTRIBUTION_IDS
METHODS, ALLOCATIONS = allocation.METHODS, allocation.ALLOCATIONS
N, ORIGINALS_PER_CELL, BUDGET_REPEATS, STREAMS, MAX_B = 5, 10, 20, 120, 4999
MAX_ELEMENTS = 1_000_000
REFERENCE_PATHS = tuple(f"evidence/rq_m09/{name}" for name in
    ("manifest.json", "hypotheses.json", "pooled_summary.csv")) + tuple(
        f"evidence/rq_m09/panels/{d}_seed{s}.npz" for d in DISTRIBUTIONS for s in (0,1,2))
NUMERIC_RTOL, NUMERIC_ATOL = allocation.NUMERIC_RTOL, allocation.NUMERIC_ATOL
IDENTITY_RTOL, IDENTITY_ATOL = allocation.IDENTITY_RTOL, allocation.IDENTITY_ATOL
prepare_sample = allocation.prepare_sample

mean_t_interval, band_status = allocation.mean_t_interval, allocation.band_status
variance_band_status, joint_status = allocation.variance_band_status, allocation.joint_status
sample_sha256 = allocation.sample_sha256


def _integer(value,name,low,high):
    return matched_moments._integer(value,name,low,high)


def _identity(distribution,n,master_seed,outer_id,namespace):
    distribution=matched_moments._distribution(distribution)
    n=_integer(n,"n",5,5);seed=_integer(master_seed,"master_seed",0,2)
    outer=_integer(outer_id,"outer_id",0,9);namespace=_integer(namespace,"namespace",0,2**32-1)
    if namespace not in (MAIN_NAMESPACE,SMOKE_NAMESPACE):raise ValueError("M10 namespace required")
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
    """Preregister all144060 identities without generating any observations."""
    namespaces={f"M{i:02d}_{kind}": (0x4D3030+i if kind=="main" else 0x534D3030+i)
                for i in range(2,10) for kind in ("main","smoke")}
    namespaces.update(M10_main=MAIN_NAMESPACE,M10_smoke=SMOKE_NAMESPACE)
    if len(set(namespaces.values()))!=len(namespaces):raise ValueError("namespace registry collision")
    originals=[dict(distribution=d,master_seed=s,outer_id=o,n=N,entropy=s,
                    spawn_key=list(original_key(d,N,s,o)))
               for d in DISTRIBUTIONS for s in (0,1,2) for o in range(10)]
    return dict(question_id="RQ-M10",generator="Generator(PCG64(SeedSequence(...)))",
        namespace_registry=namespaces,main_namespace=MAIN_NAMESPACE,smoke_namespace=SMOKE_NAMESPACE,
        distribution_ids=DISTRIBUTION_IDS.copy(),original_key_length=6,inner_key_length=7,
        originals=originals,budget_repeat_ids=list(range(20)),stream_ids=list(range(120)),
        inner_key_fields=["namespace","distribution_id","n","outer_id","purpose1","budget_repeat_id","stream_id"],
        original_count=60,inner_stream_count=144000,all_identity_count=144060,
        sample_hashes_known_before_generation=False,
        source_transform="accepted matched_moments Python-math constants and exact NumPy vector operation order; no sample standardization",
        constants={d:matched_moments.population_info(d)["constants"] for d in DISTRIBUTIONS},
        smoke_original_source="accepted M09 saved outer0 first5 only, no new M10 originals",
        excluded_from_key=["J","K","method","execution_order","batch","attempt"])


def transform_raw(raw,distribution):
    """Pure accepted float64 operation order, suitable for saved-raw readback."""
    dist=matched_moments._distribution(distribution);raw=np.asarray(raw)
    if raw.shape!=(5,) or raw.dtype.kind not in "iuf":raise ValueError("raw must be a real length5 vector")
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
    the accepted M09 reference. Real main calls are forbidden before preregister.
    """
    dist,n,seed,outer,namespace=_identity(distribution,n,master_seed,outer_id,namespace)
    if namespace!=MAIN_NAMESPACE:raise ValueError("smoke must reuse accepted M09 saved originals")
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
    except (OSError,subprocess.CalledProcessError) as exc:raise ValueError(f"accepted M09 reference unavailable: {path}") from exc
    if local!=blob:raise ValueError(f"M09 reference differs from accepted Git blob: {path}")
    return local


def _inspect_reference(root):
    root=Path(root).resolve();blobs={path:_verified_reference_blob(root,path) for path in REFERENCE_PATHS}
    try:
        manifest=json.loads(blobs["evidence/rq_m09/manifest.json"])
        hypotheses=json.loads(blobs["evidence/rq_m09/hypotheses.json"])
        pooled=list(csv.DictReader(StringIO(blobs["evidence/rq_m09/pooled_summary.csv"].decode())))
    except (ValueError,UnicodeError) as exc:raise ValueError("M09 reference JSON/CSV malformed") from exc
    if manifest.get("question_id")!="RQ-M09" or hypotheses.get("question_id")!="RQ-M09" or len(pooled)!=18:
        raise ValueError("M09 reference role/design mismatch")
    cells={}
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            path=f"evidence/rq_m09/panels/{dist}_seed{seed}.npz"
            try:
                with np.load(BytesIO(blobs[path]),allow_pickle=False) as panel:
                    for name,expected in dict(distribution=dist,master_seed=seed,n=20,O=10,R=20,
                            streams_per_budget=120,B=4999,alpha=.05,namespace=0x4D3039).items():
                        if panel[name].shape!=() or panel[name].item()!=expected:raise ValueError(f"M09 reference metadata mismatch: {name}")
                    samples,ids,hashes=panel["samples"],panel["outer_id"],panel["sample_sha256"]
                    if samples.shape!=(10,20) or samples.dtype!=np.dtype("<f8") or not np.isfinite(samples).all() or ids.dtype!=np.dtype("uint16") or not np.array_equal(ids,np.arange(10)) or hashes.shape!=(10,) or hashes.dtype!=np.dtype("S64"):
                        raise ValueError("M09 reference sample/ID/hash shape mismatch")
                    if not np.array_equal(hashes,np.array([sample_sha256(x).encode() for x in samples],dtype="S64")):
                        raise ValueError("M09 reference sample hash mismatch")
                    cell=dict(samples=samples.copy(),outer_id=ids.copy(),sample_sha256=hashes.copy())
            except (OSError,EOFError,KeyError) as exc:raise ValueError(f"M09 reference panel cannot be read: {path}") from exc
            for array in cell.values():array.flags.writeable=False
            cells[(dist,seed)]=cell
    reference=dict(schema_version=1,question_id="RQ-M10",reference_question_id="RQ-M09",
        accepted_head=ACCEPTED_HEAD,cohort_role="accepted_M09_reference_only",
        source_files_sha256={path:sha256(blob).hexdigest() for path,blob in blobs.items()},
        old_sample_count=60,selection_rule="all accepted M09 G/L x seed0,1,2 x outer0..9; smoke uses outer0 first5 only",
        new_original_hashes_preregistered=False)
    return reference,dict(cohort_role="accepted_M09_reference_only",reference=reference,
        manifest=manifest,hypotheses=hypotheses,pooled=pooled,cells=cells)


def make_reference_manifest(root=ROOT):return _inspect_reference(root)[0]


def validate_reference_manifest(reference,root=ROOT):
    if not isinstance(reference,dict) or reference!=_inspect_reference(root)[0]:raise ValueError("M09 reference manifest mismatch")


def load_reference(reference=None,root=ROOT):
    expected,loaded=_inspect_reference(root)
    if reference is not None and reference!=expected:raise ValueError("M09 reference manifest mismatch")
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
        n=np.array(5),namespace=np.array(first["namespace"]),raw_kind=np.array(first["raw_kind"]),
        constant_names=np.array(list(constants)),constant_values=np.array(list(constants.values()),dtype="<f8"))
    for array in result.values():array.flags.writeable=False
    return result


def smoke_projection(reference):
    """Six accepted n20 outer0 -> first5 samples, smoke only; never main inputs."""
    if not isinstance(reference,dict) or reference.get("cohort_role")!="accepted_M09_reference_only":
        raise ValueError("verified M09 reference required")
    cells={}
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            parent=reference["cells"][(dist,seed)]
            sample=np.asarray(parent["samples"][0,:N],dtype="<f8").copy()
            cell=dict(samples=sample[None,:],outer_id=np.array([0],dtype=np.uint16),
                sample_sha256=np.array([sample_sha256(sample).encode()],dtype="S64"),
                source_sample_sha256=np.array([parent["sample_sha256"][0]],dtype="S64"))
            for a in cell.values():a.flags.writeable=False
            cells[(dist,seed)]=cell
    return cells


def _projection_hashes(reference):
    return {sample_sha256(c["samples"][0]) for c in smoke_projection(reference).values()}


def _records_from_cells(cells):
    records={}
    for (dist,seed),cell in cells.items():
        constants=dict(zip(cell["constant_names"].astype(str),map(float,cell["constant_values"])))
        for row,outer in enumerate(cell["outer_id"]):
            records[(dist,seed,int(outer))]=dict(distribution=dist,master_seed=seed,outer_id=int(outer),
                n=N,namespace=MAIN_NAMESPACE,entropy=seed,spawn_key=[int(x) for x in cell["original_keys"][row]],
                raw_kind=cell["raw_kind"].item(),constants=constants,sample=cell["samples"][row].copy(),
                raw=cell["raw"][row].copy(),sample_sha256=cell["sample_sha256"][row].decode(),
                raw_sha256=cell["raw_sha256"][row].decode())
    return records


def generate_cohort(reference,namespace=MAIN_NAMESPACE,draw_fn=None,progress=None,existing_cells=None):
    """Guarded main only; validate preserved subset first, draw ONLY missing IDs.

    No RNG is used to load/validate existing cells. Completed and offending new
    records are preserved in progress before collision checks; never replaced.
    The runner must persist receipts and read back all60 before any inner RNG.
    """
    if namespace!=MAIN_NAMESPACE:raise ValueError("new cohort belongs to the main namespace only")
    preserved={} if existing_cells is None else existing_cells
    validate_cohort(preserved,reference,namespace,allow_partial=True)
    records=_records_from_cells(preserved);reused=len(records)
    projected=_projection_hashes(reference);seen={r["sample_sha256"] for r in records.values()}
    draw=draw_fn or draw_original;progress=progress if progress is not None else {}
    cells={k:_pack_records([r for identity,r in sorted(records.items()) if identity[:2]==k]) for k in preserved}
    completed=list(records.values())
    progress.update(cells=cells,completed_records=completed,generated_originals=0,reused_originals=reused,completed_originals=reused)
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            for outer in range(10):
                identity=(dist,seed,outer)
                if identity in records:continue
                progress.update(active_distribution=dist,active_master_seed=seed,active_outer_id=outer)
                record=draw(dist,n=N,master_seed=seed,outer_id=outer,namespace=namespace)
                records[identity]=record;completed.append(record)
                cells[(dist,seed)]=_pack_records([r for key,r in sorted(records.items()) if key[:2]==(dist,seed)])
                progress["generated_originals"]+=1;progress["completed_originals"]+=1
                digest=sample_sha256(record["sample"])
                if digest in projected:raise ValueError("unexpected exact smoke projection match; no replacement")
                if digest in seen:raise ValueError("unexpected duplicate new-cohort sample; no replacement")
                seen.add(digest)
    validate_cohort(cells,reference,namespace)
    return cells


def validate_cohort(cells,reference,namespace=MAIN_NAMESPACE,allow_partial=False):
    """Read-only full or preserved-subset validation; no original generation."""
    if namespace!=MAIN_NAMESPACE:raise ValueError("new cohort main namespace required")
    if not isinstance(allow_partial,(bool,np.bool_)):raise ValueError("allow_partial must be boolean")
    expected={(d,s) for d in DISTRIBUTIONS for s in (0,1,2)}
    if not isinstance(cells,dict) or not set(cells)<=expected or (not allow_partial and set(cells)!=expected):
        raise ValueError("cohort fixed-cell set mismatch")
    projected,seen=_projection_hashes(reference),set();count=0;identities=[]
    for dist,seed in sorted(cells):
        cell=cells[(dist,seed)]
        required={"samples","raw","sample_sha256","raw_sha256","outer_id","original_keys",
                  "distribution","master_seed","n","namespace","raw_kind","constant_names","constant_values"}
        if not isinstance(cell,dict) or not required<=cell.keys():raise ValueError("new cohort raw/key metadata required; smoke/reference cells forbidden")
        def scalar(name):
            value=np.asarray(cell[name])
            if value.shape!=():raise ValueError(f"cohort metadata must be scalar: {name}")
            return value.item()
        for name,value in dict(distribution=dist,master_seed=seed,n=N,namespace=namespace,
                               raw_kind="gamma_y" if dist=="gamma" else "normal_z").items():
            if scalar(name)!=value:raise ValueError(f"cohort metadata mismatch: {name}")
        constants=matched_moments.population_info(dist)["constants"]
        if list(np.asarray(cell["constant_names"]).astype(str))!=list(constants) or np.asarray(cell["constant_values"]).dtype!=np.dtype("<f8") or np.asarray(cell["constant_values"]).tobytes()!=np.asarray(list(constants.values()),dtype="<f8").tobytes():
            raise ValueError("cohort analytic constants differ")
        sample,raw=np.asarray(cell["samples"]),np.asarray(cell["raw"])
        ids,keys=np.asarray(cell["outer_id"]),np.asarray(cell["original_keys"]);size=len(ids) if ids.ndim==1 else -1
        if size<1 or size>10 or sample.shape!=(size,N) or raw.shape!=(size,N) or sample.dtype!=np.dtype("<f8") or raw.dtype!=np.dtype("<f8") or not np.isfinite(sample).all() or not np.isfinite(raw).all():
            raise ValueError("cohort raw/sample shape/dtype/nonfinite mismatch")
        if ids.dtype!=np.dtype("uint16") or len(set(map(int,ids)))!=size or not np.all(ids<10) or not np.array_equal(ids,np.sort(ids)) or (not allow_partial and not np.array_equal(ids,np.arange(10))) or keys.shape!=(size,6) or keys.dtype!=np.dtype("uint32"):
            raise ValueError("cohort ID/key shape/dtype mismatch")
        for name in ("sample_sha256","raw_sha256"):
            if np.asarray(cell[name]).shape!=(size,) or np.asarray(cell[name]).dtype!=np.dtype("S64"):raise ValueError("cohort hash shape/dtype mismatch")
        for row,outer in enumerate(ids):
            if tuple(keys[row])!=original_key(dist,N,seed,int(outer),namespace):raise ValueError("original six-element key mismatch")
            transformed=transform_raw(raw[row],dist)
            if transformed.tobytes()!=sample[row].tobytes():raise ValueError("saved raw transform/sample bytes mismatch")
            digest=sample_sha256(sample[row]);raw_digest=sample_sha256(raw[row])
            if digest.encode()!=cell["sample_sha256"][row] or raw_digest.encode()!=cell["raw_sha256"][row]:raise ValueError("saved hash mismatch")
            if digest in projected:raise ValueError("unexpected exact smoke projection match; no replacement")
            if digest in seen:raise ValueError("unexpected duplicate new-cohort sample; no replacement")
            seen.add(digest);identities.append(dict(distribution=dist,master_seed=seed,outer_id=int(outer)));count+=1
    return dict(original_samples=count,unique_new_sample_hashes=len(seen),exact_smoke_projection_matches=0,
        duplicate_new_samples=0,raw_transform_bytes_verified=True,complete=count==60,retained_ids=identities,
        interpretation="key separation and no exact smoke-projection match are implementation checks, not a proof of statistical independence")


def cohort_receipt(cells,reference,source_commit,namespace=MAIN_NAMESPACE,partial=False):
    if not isinstance(source_commit,str) or len(source_commit)!=40 or any(c not in "0123456789abcdef" for c in source_commit):raise ValueError("full source commit required")
    if not isinstance(partial,(bool,np.bool_)):raise ValueError("partial must be boolean")
    summary=validate_cohort(cells,reference,namespace,allow_partial=partial)
    records=list(_records_from_cells(cells).values())
    records=[{k:v for k,v in r.items() if k not in ("sample","raw")} for r in records]
    records.sort(key=lambda r:(r["distribution"],r["master_seed"],r["outer_id"]))
    return dict(schema_version=1,question_id="RQ-M10",source_git_sha=source_commit,cohort_role="new_M10_main_originals",
        status="partial" if partial and len(records)<60 else "complete",namespace=namespace,
        hash_recording_phase="after generation inside the verified main attempt, never preregistered before creation",
        original_samples=len(records),records=records,isolation_checks=summary,
        M09_reference_accepted_head=ACCEPTED_HEAD,M09_reference_files_sha256=reference.get("reference",{}).get("source_files_sha256",{}))


def load_cohort(directory,reference,namespace=MAIN_NAMESPACE,allow_partial=False):
    cells={};directory=Path(directory)
    expected={f"{d}_seed{s}.npz":(d,s) for d in DISTRIBUTIONS for s in (0,1,2)}
    actual={file.name for file in directory.glob("*.npz")}
    if actual-set(expected):raise ValueError("unexpected original cell file")
    for name in sorted(actual):
        with np.load(directory/name,allow_pickle=False) as panel:cell={k:panel[k] for k in panel.files}
        for array in cell.values():array.flags.writeable=False
        cells[expected[name]]=cell
    validate_cohort(cells,reference,namespace,allow_partial=allow_partial)
    return cells


def generate_bootstrap_means(sample,distribution,n=N,master_seed=0,outer_id=0,
        budget_repeat_id=0,stream_id=0,Bmax=MAX_B,namespace=MAIN_NAMESPACE,
        max_elements=MAX_ELEMENTS,return_indices=False):
    raw=np.asarray(sample)
    if raw.ndim!=1 or raw.size!=n or raw.dtype.kind not in "iuf":raise ValueError("sample must be real length5 vector")
    sample=raw.astype(float)
    if not np.isfinite(sample).all() or np.all(sample==sample[0]):raise ValueError("sample is nonfinite or constant")
    Bmax=_integer(Bmax,"Bmax",1,MAX_B);cap=_integer(max_elements,"max_elements",N,MAX_ELEMENTS)
    if not isinstance(return_indices,(bool,np.bool_)):raise ValueError("return_indices must be boolean")
    rng=rng_for_panel(distribution,n,master_seed,outer_id,1,budget_repeat_id,stream_id,namespace)
    means=np.empty(Bmax);saved=np.empty((Bmax,N),np.uint8) if return_indices else None
    for start in range(0,Bmax,cap//N):
        count=min(cap//N,Bmax-start);indices=rng.integers(0,N,size=(count,N),dtype=np.int64)
        with np.errstate(over="ignore",invalid="ignore"):means[start:start+count]=sample[indices].mean(axis=1)
        if return_indices:saved[start:start+count]=indices.astype(np.uint8)
    if not np.isfinite(means).all():raise ValueError("bootstrap_mean_nonfinite")
    return (means,saved) if return_indices else means


UNKNOWN_COUNT=65535


def tie_diagnostics(context,means):
    """Actual comparison counts used by unchanged half-tie z0; no new gate."""
    try:values=np.asarray(means)
    except (TypeError,ValueError):values=np.asarray(())
    available=(isinstance(context,dict) and bool(context.get("valid",False))
        and np.isfinite(context.get("mean",np.nan)) and values.ndim==1 and values.dtype.kind in "iuf"
        and 1<=values.size<=MAX_B and np.isfinite(values).all())
    if not available:return dict(B=int(values.size) if values.ndim==1 else 0,less_count=UNKNOWN_COUNT,equal_count=UNKNOWN_COUNT,
        tie_count=UNKNOWN_COUNT,greater_count=UNKNOWN_COUNT,bias_probability=float("nan"))
    less=int(np.count_nonzero(values<context["mean"]));equal=int(np.count_nonzero(values==context["mean"]));greater=values.size-less-equal
    return dict(B=int(values.size),less_count=less,equal_count=equal,tie_count=equal,greater_count=int(greater),
        bias_probability=(less+.5*equal)/values.size)


def infer_single(context,means):
    """Unchanged M09 B4999 inference, plus non-gating integer tie diagnostics."""
    records=allocation.infer_single(context,means);counts=tie_diagnostics(context,means)
    return {method:{**record,**{k:counts[k] for k in ("less_count","equal_count")}} for method,record in records.items()}


def budget_metrics(data,se,group_size=30):
    result=allocation.budget_metrics(data,se,group_size)
    result["allocations"]["Apop"]*=.5  # sqrt(5)/sqrt(20); all other math stays accepted.
    return result


def original_metrics(data,se,group_size=30,expected_repeats=BUDGET_REPEATS):
    result=allocation.original_metrics(data,se,group_size,expected_repeats)
    result["budget"]["allocations"]["Apop"]*=.5
    result["original_allocations"]["Apop"]*=.5
    return result


def summarize(rows,pooled=False):
    result=accepted_runner.summarize(rows,pooled=pooled)
    return [{**r,"n":N} for r in result]


def assemble_hypotheses(comparisons):
    return {**accepted_runner.assemble_hypotheses(comparisons),"question_id":"RQ-M10","sample_n":N}


def decomposition_summary(comparisons,decompositions):
    return accepted_runner.decomposition_summary(comparisons,decompositions)
