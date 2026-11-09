"""RQ-M08: shared resampling budget, average group SD, repeated-budget variance.

All three allocations use the very same120 streams. T averages square roots
within each nonoverlapping partition; it never takes a root after pooling V.
"""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess

import numpy as np

from mathscape import finite_k as accepted

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_HEAD = "39bcdc367320b6da513df72d2fa43bfdb7f060f4"
M07_MANIFEST_PATH = "evidence/rq_m07/manifest.json"
DISTRIBUTIONS, DISTRIBUTION_IDS = ("gamma", "lognormal"), {"gamma": 1, "lognormal": 2}
METHODS = ("t", "percentile", "bca")
ALLOCATIONS, ALLOCATION_J = ("P30", "P60", "P120"), (4, 2, 1)
GROUP_ALLOCATION = (0, 0, 0, 0, 1, 1, 2)
N, ORIGINALS_PER_CELL, BUDGET_REPEATS, STREAMS, MAX_B = 20, 10, 20, 120, 4999
MAIN_NAMESPACE, SMOKE_NAMESPACE = 0x4D3038, 0x534D3038
MAX_ELEMENTS = 1_000_000
NUMERIC_RTOL, NUMERIC_ATOL = accepted.NUMERIC_RTOL, accepted.NUMERIC_ATOL
IDENTITY_RTOL, IDENTITY_ATOL = accepted.IDENTITY_RTOL, accepted.IDENTITY_ATOL
CONFIDENCE = .995
INPUT_PATHS = (M07_MANIFEST_PATH,) + tuple(
    f"evidence/rq_m07/panels/{d}_seed{s}.npz" for d in DISTRIBUTIONS for s in (0, 1, 2))
ALLOCATION_METRICS = ("T", "Vbar", "lower_sd", "upper_sd", "S", "Ase", "Apop", "Vse",
                      "C", "Mbelow", "Mabove", "miss_T", "mean_width", "warning_rate")
GROUP_FIELDS = ("lower_mean", "upper_mean", "lower_var", "upper_var", "V", "A")
DECOMPOSITION_FIELDS = ("Vbar30", "T30", "T60", "T120", "Jgap", "Pgap", "delta",
    "residual", "normalized_residual", "abs_normalized_residual") + tuple(
        f"ss{partition}_{endpoint}_{field}" for partition in (30, 60)
        for endpoint in ("lower", "upper") for field in ("residual", "scale", "normalized_residual", "abs_normalized_residual"))
prepare_sample, infer_single = accepted.prepare_sample, accepted.infer_single
sample_sha256, band_status, joint_status = accepted.sample_sha256, accepted.band_status, accepted.joint_status


def mean_t_interval(values, confidence=CONFIDENCE):
    """Original-unit interval; default99.5%, never budget-repeat pseudoreplicates."""
    return accepted.mean_t_interval(values, confidence=confidence)


def _verified_blob(root, path):
    try:
        local = (root/path).read_bytes()
        blob = subprocess.run(["git", "show", f"{ACCEPTED_HEAD}:{path}"], cwd=root,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"accepted M07 input unavailable: {path}") from exc
    if local != blob:
        raise ValueError(f"saved M07 input differs from accepted Git blob: {path}")
    return local


def _inspect_inputs(root):
    root = Path(root).resolve()
    blobs = {path: _verified_blob(root, path) for path in INPUT_PATHS}
    try:
        source = json.loads(blobs[M07_MANIFEST_PATH])
    except (ValueError, UnicodeError) as exc:
        raise ValueError("M07 manifest JSON is invalid") from exc
    expected = dict(distributions=list(DISTRIBUTIONS), n=20, master_seeds=[0, 1, 2],
                    R=10, J=20, K=120, K_first=30, groups=4, B=4999, alpha=.05,
                    outer_ids=list(range(10)), methods=list(METHODS), input_accepted_head=accepted.ACCEPTED_HEAD)
    if source.get("question_id") != "RQ-M07" or any(source.get("design", {}).get(k) != v for k,v in expected.items()):
        raise ValueError("M07 input design mismatch")
    # Accepted M07 itself recorded this exact M06/M05 Git/blob/hash lineage.
    lineage, upstream_cells = accepted._inspect_inputs(root)
    if source.get("inputs") != lineage:
        raise ValueError("M07/M06/M05 accepted lineage differs")
    cells, selections = {}, []
    for dist in DISTRIBUTIONS:
        for seed in (0, 1, 2):
            path = f"evidence/rq_m07/panels/{dist}_seed{seed}.npz"
            try:
                with np.load(BytesIO(blobs[path]), allow_pickle=False) as panel:
                    for key,value in dict(distribution=dist,master_seed=seed,n=20,R=10,
                        observed_original_count=10,J=20,K=120,K_first=30,B=4999,alpha=.05,
                        namespace=0x4D3037,input_accepted_head=accepted.ACCEPTED_HEAD).items():
                        accepted._scalar(panel,key,value)
                    if not np.array_equal(panel["methods"], METHODS):
                        raise ValueError("M07 method metadata mismatch")
                    sample,ids,hashes = panel["samples"],panel["outer_id"],panel["sample_sha256"]
                    if sample.shape!=(10,20) or sample.dtype!=np.dtype("<f8") or not np.isfinite(sample).all():
                        raise ValueError("M07 saved original shape/dtype/nonfinite mismatch")
                    if ids.dtype!=np.dtype("uint16") or not np.array_equal(ids,np.arange(10)) or hashes.shape!=(10,) or hashes.dtype!=np.dtype("S64"):
                        raise ValueError("M07 original ID/hash metadata mismatch")
                    old=upstream_cells[(dist,seed)]
                    if sample.astype("<f8").tobytes()!=old["samples"].astype("<f8").tobytes() or not np.array_equal(hashes,old["sample_sha256"]):
                        raise ValueError("M07 original bytes/hash differ from M06/M05")
                    if [sample_sha256(x).encode() for x in sample]!=list(hashes):
                        raise ValueError("M07 original sample hash mismatch")
                    cell=dict(samples=sample.copy(),outer_id=ids.copy(),sample_sha256=hashes.copy())
            except (OSError,EOFError,KeyError) as exc:
                raise ValueError(f"M07 saved input cannot be read: {path}") from exc
            for array in cell.values():array.flags.writeable=False
            cells[(dist,seed)]=cell
            selections.append(dict(distribution=dist,master_seed=seed,source_path=path,
                m06_source_path=f"evidence/rq_m06/panels/{dist}_seed{seed}.npz",
                m05_source_path=f"evidence/rq_m05/panels/{dist}_seed{seed}.npz",outer_ids=list(range(10)),
                sample_sha256=[h.decode() for h in hashes],selected_samples_sha256=sample_sha256(sample)))
    manifest=dict(schema_version=1,question_id="RQ-M08",accepted_head=ACCEPTED_HEAD,
        source_question_id="RQ-M07",source_files_sha256={path:sha256(blob).hexdigest() for path,blob in blobs.items()},
        m06_accepted_head=accepted.ACCEPTED_HEAD,m06_files_sha256=lineage["source_files_sha256"],
        m05_accepted_head=lineage["upstream_accepted_head"],m05_files_sha256=lineage["upstream_files_sha256"],
        n=20,originals_per_cell=10,selected_originals=60,
        selection_rule="same exact M07 gamma/lognormal x seeds0,1,2 x saved outer IDs0..9, no endpoint reuse or outcome selection",
        sample_hash_contract="SHA256 little-endian float64 C-order bytes",selections=selections)
    return manifest,cells


def make_inputs_manifest(root=ROOT):return _inspect_inputs(root)[0]


def validate_inputs_manifest(inputs, root=ROOT):
    if not isinstance(inputs,dict) or inputs!=_inspect_inputs(root)[0]:
        raise ValueError("M08 input manifest differs from verified saved originals")


def load_inputs(inputs=None, root=ROOT):
    expected,cells=_inspect_inputs(root)
    if inputs is not None and (not isinstance(inputs,dict) or inputs!=expected):
        raise ValueError("M08 input manifest differs from verified saved originals")
    return cells


def rng_for_panel(distribution,n=N,master_seed=0,outer_id=0,purpose_id=1,
                  budget_repeat_id=0,stream_id=0,namespace=MAIN_NAMESPACE):
    dist=accepted._distribution(distribution)
    n=accepted._integer(n,"n",20,20);seed=accepted._integer(master_seed,"master_seed",0,2)
    outer=accepted._integer(outer_id,"outer_id",0,9);purpose=accepted._integer(purpose_id,"purpose_id",1,1)
    repeat=accepted._integer(budget_repeat_id,"budget_repeat_id",0,19)
    stream=accepted._integer(stream_id,"stream_id",0,119)
    namespace=accepted._integer(namespace,"namespace",0,2**32-1)
    if namespace not in (MAIN_NAMESPACE,SMOKE_NAMESPACE):raise ValueError("namespace must be M08 main or smoke")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=seed,
        spawn_key=(namespace,DISTRIBUTION_IDS[dist],n,outer,purpose,repeat,stream))))


def generate_bootstrap_means(sample,distribution,n=N,master_seed=0,outer_id=0,
        budget_repeat_id=0,stream_id=0,Bmax=MAX_B,namespace=MAIN_NAMESPACE,
        max_elements=MAX_ELEMENTS,return_indices=False):
    rng=rng_for_panel(distribution,n,master_seed,outer_id,1,budget_repeat_id,stream_id,namespace)
    raw=np.asarray(sample)
    if raw.ndim!=1 or raw.size!=n or raw.dtype.kind not in "iuf":raise ValueError("sample must be real length20 vector")
    sample=raw.astype(float)
    if not np.isfinite(sample).all() or np.all(sample==sample[0]):raise ValueError("sample is nonfinite or constant")
    Bmax=accepted._integer(Bmax,"Bmax",1,MAX_B)
    cap=accepted._integer(max_elements,"max_elements",20,MAX_ELEMENTS)
    if not isinstance(return_indices,(bool,np.bool_)):raise ValueError("return_indices must be boolean")
    means=np.empty(Bmax);saved=np.empty((Bmax,20),np.uint8) if return_indices else None
    for start in range(0,Bmax,cap//20):
        count=min(cap//20,Bmax-start)
        indices=rng.integers(0,20,size=(count,20),dtype=np.int64)
        with np.errstate(over="ignore",invalid="ignore"):means[start:start+count]=sample[indices].mean(axis=1)
        if return_indices:saved[start:start+count]=indices.astype(np.uint8)
    if not np.isfinite(means).all():raise ValueError("bootstrap_mean_nonfinite")
    return (means,saved) if return_indices else means


def group_slices(group_size=30):
    g=accepted._integer(group_size,"group_size",3,30)
    return ((0,g),(g,2*g),(2*g,3*g),(3*g,4*g),(0,2*g),(2*g,4*g),(0,4*g))


def _empty_layer(shape,fields):
    out={key:np.full(shape,np.nan) for key in fields}
    out.update(complete=np.zeros(shape,bool),total_streams=np.zeros(shape,int),
               valid_streams=np.zeros(shape,int),invalid_streams=np.zeros(shape,int))
    return out


def budget_metrics(data,se,group_size=30):
    """One120-stream budget; groups(method,7), allocations(allocation,method).

    All allocations require all120 intervals for that method. Descriptive SDs
    and V are averages of group quantities, while T averages group square roots.
    C/miss/width/warning use the shared120 intervals; miss_T is coverage direction,
    distinct from the budget estimator T. Known population SE is evaluation only.
    """
    slices=group_slices(group_size);g=group_size
    lo,hi,valid,warning=arrays=accepted._endpoint_arrays(data,2)
    if lo.shape!=(3,4*g):raise ValueError("budget must contain exactly4*group_size streams")
    accepted._t_invariant(arrays)
    groups=_empty_layer((3,7),GROUP_FIELDS)
    alloc=_empty_layer((3,3),ALLOCATION_METRICS)
    decomp=_empty_layer((3,),DECOMPOSITION_FIELDS)
    decomp.update(algebra_ok=np.zeros(3,bool),reason=np.full(3,"",dtype="U80"))
    for mi,method in enumerate(METHODS):
        l,u=lo[mi],hi[mi]
        with np.errstate(over="ignore",invalid="ignore"):width=u-l
        mask=valid[mi]&np.isfinite(l)&np.isfinite(u)&(l<=u)&np.isfinite(width)
        total,count=(1,int(mask.all())) if method=="t" else (4*g,int(mask.sum()))
        for layer in (alloc,decomp):
            if layer is alloc:layer["total_streams"][:,mi],layer["valid_streams"][:,mi],layer["invalid_streams"][:,mi]=total,count,total-count
            else:layer["total_streams"][mi],layer["valid_streams"][mi],layer["invalid_streams"][mi]=total,count,total-count
        for gi,(start,stop) in enumerate(slices):
            gt,gc=(1,int(mask.all())) if method=="t" else (stop-start,int(mask[start:stop].sum()))
            groups["total_streams"][mi,gi],groups["valid_streams"][mi,gi],groups["invalid_streams"][mi,gi]=gt,gc,gt-gc
        if not accepted._valid_scale(se) or not mask.all():
            decomp["reason"][mi]="sample_se_invalid" if not accepted._valid_scale(se) else "endpoint_invalid_or_nonfinite"
            continue
        with np.errstate(over="ignore",invalid="ignore",divide="ignore",under="ignore"):
            lm=np.array([l[a:b].mean() for a,b in slices]);um=np.array([u[a:b].mean() for a,b in slices])
            lv=np.array([accepted._variance(l[a:b]) for a,b in slices]);uv=np.array([accepted._variance(u[a:b]) for a,b in slices])
            V=(lv+uv)/(2*np.float64(se)*np.float64(se));A=np.sqrt(V)
            if method=="t":lv[:]=0;uv[:]=0;V[:]=0;A[:]=0
            av={};selection=((0,1,2,3),(4,5),(6,))
            for pi,index in enumerate(selection):
                index=list(index);ls=np.sqrt(lv[index]);us=np.sqrt(uv[index]);s=np.hypot(ls,us)/np.sqrt(2)
                av[pi]=dict(T=A[index].mean(),Vbar=V[index].mean(),lower_sd=ls.mean(),upper_sd=us.mean(),S=s.mean(),
                    Ase=A[index].mean(),Apop=s.mean()*np.sqrt(20),Vse=V[index].mean(),
                    C=np.mean((l<=0)&(0<=u)),Mbelow=np.mean(u<0),Mabove=np.mean(l>0),miss_T=np.mean(u<0)-np.mean(l>0),
                    mean_width=width.mean(),warning_rate=warning[mi].mean())
            vbar=V[:4].mean();t30,t60,t120=[av[pi]["T"] for pi in range(3)]
            jgap,pgap=np.sqrt(vbar)-t30,t120-np.sqrt(vbar)
            delta=t120-t30;residual=delta-(jgap+pgap)
            values=dict(Vbar30=vbar,T30=t30,T60=t60,T120=t120,Jgap=jgap,Pgap=pgap,delta=delta,
                residual=residual,normalized_residual=residual/max(1,abs(delta),abs(jgap+pgap)),
                abs_normalized_residual=abs(residual/max(1,abs(delta),abs(jgap+pgap))))
            ss_ok=True
            for partition,index in ((30,(0,1,2,3)),(60,(4,5))):
                size=g if partition==30 else 2*g
                for name,means,variances,x in (("lower",lm,lv,l),("upper",um,uv,u)):
                    lhs=(4*g-1)*variances[6]
                    rhs=(size-1)*variances[list(index)].sum()+size*np.sum((means[list(index)]-x.mean())**2)
                    raw=lhs-rhs;scale=max(abs(lhs),abs(rhs));norm=raw/max(1,scale)
                    if method=="t":raw,scale,norm=0.,0.,0.
                    prefix=f"ss{partition}_{name}_"
                    values.update({prefix+"residual":raw,prefix+"scale":scale,prefix+"normalized_residual":norm,prefix+"abs_normalized_residual":abs(norm)})
                    ss_ok=ss_ok and abs(raw)<=IDENTITY_ATOL+IDENTITY_RTOL*scale
            if method=="t":
                for key in DECOMPOSITION_FIELDS:values[key]=0.
            finite=np.isfinite([*lm,*um,*lv,*uv,*V,*A,*values.values()]).all() and all(np.isfinite(list(x.values())).all() for x in av.values())
        if not finite:
            decomp["reason"][mi]="derived_statistic_nonfinite";continue
        for key,value in values.items():decomp[key][mi]=value
        algebra_ok=ss_ok and jgap>=-(IDENTITY_ATOL+IDENTITY_RTOL*max(abs(np.sqrt(vbar)),abs(t30))) and abs(residual)<=IDENTITY_ATOL+IDENTITY_RTOL*max(abs(delta),abs(jgap+pgap))
        decomp["algebra_ok"][mi]=algebra_ok
        if not algebra_ok:
            decomp["reason"][mi]="algebraic_identity_failed";continue
        for key,value in dict(lower_mean=lm,upper_mean=um,lower_var=lv,upper_var=uv,V=V,A=A).items():groups[key][mi]=value
        groups["complete"][mi]=True;alloc["complete"][:,mi]=True;decomp["complete"][mi]=True
        for pi,value in av.items():
            for key,number in value.items():alloc[key][pi,mi]=number
    return dict(groups=groups,allocations=alloc,decomposition=decomp)


def original_metrics(data,se,group_size=30,expected_repeats=BUDGET_REPEATS):
    """One original, allR budgets; method-local completeness and paired d/L/U."""
    repeats=accepted._integer(expected_repeats,"expected_repeats",2,BUDGET_REPEATS)
    lo,hi,valid,warning=arrays=accepted._endpoint_arrays(data,3)
    if lo.shape[0]!=repeats:raise ValueError("all planned budget-repeat slots required, including missing slots")
    accepted._t_invariant(arrays)
    result=[budget_metrics(dict(lower=lo[r],upper=hi[r],valid=valid[r],tail_warning=warning[r]),se,group_size) for r in range(repeats)]
    budget={layer:{key:np.stack([r[layer][key] for r in result]) for key in result[0][layer]} for layer in result[0]}
    oa=_empty_layer((3,3),("M","Q","Q_over_R",*ALLOCATION_METRICS))
    oa.update(complete_repeats=budget["allocations"]["complete"].sum(axis=0),planned_repeats=np.full((3,3),repeats,int))
    fields=("d","L","U","Q30","Q120","M30","M120","ratio_Q120_Q30","mean_Jgap","mean_Pgap","mean_delta",
            "max_abs_residual","max_abs_normalized_residual")+tuple(f"max_abs_ss{p}_{e}_{field}" for p in (30,60) for e in ("lower","upper") for field in ("residual","normalized_residual"))
    oc=_empty_layer((3,),fields)
    oc.update(complete_repeats=budget["decomposition"]["complete"].sum(axis=0),planned_repeats=np.full(3,repeats,int))
    for layer,source in ((oa,budget["allocations"]),(oc,budget["decomposition"])):
        for key in ("total_streams","valid_streams","invalid_streams"):layer[key][...]=source[key].sum(axis=0)
        if layer is oa:
            layer["total_streams"][:,0]=1;layer["valid_streams"][:,0]=np.all(source["valid_streams"][:,:,0]==1,axis=0).astype(int);layer["invalid_streams"][:,0]=1-layer["valid_streams"][:,0]
        else:
            layer["total_streams"][0]=1;layer["valid_streams"][0]=int(np.all(source["valid_streams"][:,0]==1));layer["invalid_streams"][0]=1-layer["valid_streams"][0]
    for mi in range(3):
        if not budget["decomposition"]["complete"][:,mi].all():continue
        with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
            av={pi:{key:float(budget["allocations"][key][:,pi,mi].mean()) for key in ALLOCATION_METRICS} for pi in range(3)}
            for pi in range(3):
                series=budget["allocations"]["T"][:,pi,mi]
                Q=0. if mi==0 else accepted._variance(series)
                av[pi].update(M=float(series.mean()),Q=Q,Q_over_R=Q/repeats)
            d=float(np.mean(budget["allocations"]["T"][:,2,mi]-budget["allocations"]["T"][:,0,mi]))
            values=dict(d=d,L=av[2]["Q"]-.8*av[0]["Q"],U=av[2]["Q"]-1.25*av[0]["Q"],
                Q30=av[0]["Q"],Q120=av[2]["Q"],M30=av[0]["M"],M120=av[2]["M"])
            for key in ("Jgap","Pgap","delta"):values["mean_"+key]=float(budget["decomposition"][key][:,mi].mean())
            for key in fields:
                if key.startswith("max_abs_"):
                    values[key]=float(np.max(np.abs(budget["decomposition"][key.removeprefix("max_abs_")][:,mi])))
            ratio=av[2]["Q"]/av[0]["Q"] if av[0]["Q"]>0 else np.nan
        if not all(np.isfinite(list(x.values())).all() for x in av.values()) or not np.isfinite(list(values.values())).all():continue
        oa["complete"][:,mi]=True;oc["complete"][mi]=True
        for pi,value in av.items():
            for key,number in value.items():oa[key][pi,mi]=number
        for key,number in values.items():oc[key][mi]=number
        oc["ratio_Q120_Q30"][mi]=ratio if np.isfinite(ratio) else np.nan
    return dict(budget=budget,original_allocations=oa,original_comparisons=oc)


def variance_band_status(L_low,L_high,U_low,U_high,mean_Q30,complete=True):
    """H2 joint inequalities, only after BOTH intervals/denominator are valid.

    Caller complete includes full original N and both interval validity flags.
    Undefined or SE0 necessary comparisons take priority over contrary evidence.
    """
    base=dict(status="inconclusive",reason="incomplete_or_undefined_primary_comparison")
    try:finite=np.isfinite([L_low,L_high,U_low,U_high,mean_Q30]).all()
    except (TypeError,ValueError):finite=False
    if not complete or not finite or mean_Q30<=0 or L_low>=L_high or U_low>=U_high:return base
    if L_low>0 and U_high<0:return dict(status="supported",reason="")
    if L_high<=0 or U_low>=0:return dict(status="not_supported",reason="")
    return dict(status="inconclusive",reason="multiplicative_band_touched_or_crossed")
