"""Preregister and run the sole RQ-M07 saved-original finite-K experiment.

No mother-sample generation exists here. Root launches main via the guard.
Large derived tables stream to CSV.GZ; only24 fixed streams retain indices.
"""
from time import perf_counter
_PROCESS_START=perf_counter()

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime,timezone
import gzip
import io
import json
from importlib.metadata import version
from pathlib import Path
import signal
import subprocess
import sys


class BudgetExceeded(RuntimeError):
    pass


def _deadline(signum,frame):
    raise BudgetExceeded("294-second internal soft stop; reserve cleanup before outer deadline")


if __name__=="__main__" and ("--run" in sys.argv or "--smoke" in sys.argv):
    signal.signal(signal.SIGALRM,_deadline);signal.signal(signal.SIGTERM,_deadline)
    signal.setitimer(signal.ITIMER_REAL,max(.001,294-(perf_counter()-_PROCESS_START)))

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from mathscape.finite_k import (
    make_inputs_manifest,validate_inputs_manifest,load_inputs,generate_bootstrap_means,
    prepare_sample,infer_single,original_metrics,mean_t_interval,band_status,reduction_status,joint_status,
    K_METRICS,GROUP_FIELDS,REPLICATE_FIELDS,NUMERIC_RTOL,NUMERIC_ATOL,IDENTITY_RTOL,IDENTITY_ATOL,
)
from scripts.run_bootstrap_benchmark import environment,sha256,peak_rss_bytes,directory_bytes,_write_json,write_csv
from scripts.run_stability_benchmark import sample_hash

DISTRIBUTIONS=("gamma","lognormal")
METHODS=("t","percentile","bca")
MAIN_NAMESPACE,SMOKE_NAMESPACE=0x4D3037,0x534D3037
ACCEPTED_HEAD="e2feae23397d014f6aa3363773da9913da85fcf7"
DESIGN=dict(distributions=list(DISTRIBUTIONS),n=20,master_seeds=[0,1,2],R=10,J=20,K=120,K_first=30,groups=4,B=4999,alpha=.05,
            outer_ids=list(range(10)),methods=list(METHODS),input_accepted_head=ACCEPTED_HEAD,source_question="RQ-M06")
POLICY=dict(wall_seconds=300,internal_stop_seconds=294,cleanup_reserve_seconds=6,peak_rss_bytes=2*1024**3,
            output_bytes=50*1024**2,max_resample_elements=1000000,decision_confidence=.99,confirmatory_intervals=4,
            public_overhead_reserve_bytes=4*1024**2,cell_cleanup_reserve_bytes=8*1024**2,
            mean_movement_margin=.0025,variance_reference_multiplier=.5)
SOURCES=("mathscape/finite_k.py","mathscape/matched_stability.py","mathscape/matched_moments.py","mathscape/stability.py","mathscape/bootstrap.py","mathscape/inference.py",
         "scripts/run_finite_k_benchmark.py","scripts/run_finite_k_guarded.py","scripts/audit_finite_k.py","scripts/run_bootstrap_benchmark.py","scripts/run_stability_benchmark.py",
         "tests/test_finite_k.py","tests/test_finite_k_runner.py","tests/test_finite_k_guard.py","tests/test_finite_k_audit.py","requirements.txt","evidence/rq_m07/SPEC.md")
DIAGNOSTICS=("lower","upper","z0","a","q_low","q_high","tau_low","tau_high")
SUMMARY_METRICS=("A","V","Q",*K_METRICS)
GAPS=("Jgap","Pgap","Rgap","delta")
RESIDUALS=("decomposition_residual","lower_ss_residual","upper_ss_residual",
           "decomposition_normalized_residual","lower_ss_normalized_residual","upper_ss_normalized_residual")
QUANTILES=(.05,.25,.50,.75,.95)


def runtime_environment():
    return {**environment(),"streamlit":version("streamlit"),"pytest":version("pytest")}


def fixed_cells():
    return [(dist,seed) for dist in DISTRIBUTIONS for seed in (0,1,2)]


def make_manifest():
    return dict(schema_version=1,question_id="RQ-M07",created_utc=datetime.now(timezone.utc).isoformat(),design=DESIGN,inputs=make_inputs_manifest(root=ROOT),
        totals=dict(original_samples=60,originals_per_population=30,input_cells=6,inner_streams=144000,bootstrap_samples=719856000,
                    logical_resampling_indices=14397120000,logical_intervals=432000,original_rows=180,original_K_rows=360,
                    replicate_rows=3600,group_rows=14400,K_rows=7200,seed_summary_rows=36,pooled_summary_rows=12),
        input_contract="accepted M06 saved G/L seed0,1,2 outer0..9, exactbytes/hash/IDs and M05 provenance verified; no result-based selection, mother generation, sample standardization or mixing old K30 streams",
        disclosure="M06 aggregate results were known at design; reused originals are not unused independent replication and do not retest/update M05/M06 conclusions",
        randomness=dict(generator="Generator(PCG64(SeedSequence(...)))",entropy="saved master_seed",main_namespace=MAIN_NAMESPACE,smoke_namespace=SMOKE_NAMESPACE,
                        distribution_ids=dict(gamma=1,lognormal=2),spawn_key=["namespace","distribution_id","n","outer_id","purpose_id","replicate_id","inner_id"],
                        purpose_ids=dict(bootstrap=1),replicate_ids=list(range(20)),inner_ids=list(range(120)),outer_ids=list(range(10)),native_indices_dtype="int64",
                        excluded_from_key=["B","K","method","execution_order","batch_size"],
                        sharing="same original/r/k mean vector shared by percentile and BCa; K30 is first k0..29 of same r's K120; r/k streams independent; prepare t/a/se only once per original"),
        inference="unchanged accepted t/ddof1,linear quantile,half-tie z0,jackknife a/BCa validity; truth/population labels not passed to inference, no clipping/fallback/adaptiveB; tau<10 retained warnings",
        metrics=dict(V="(sample_var(lower)+sample_var(upper))/(2*sample_SE^2),ddof1",A="sqrt(V)",K30="always first group0,never 4-group average",
                     delta="A120-A30_first per r",d="mean_r(delta) per original",Q30="sample_var_r(A30_first),ddof1",Q120="sample_var_r(A120),ddof1",W="Q120-.5*Q30",
                     decomposition="Vbar=mean4(Vgroup);Abar=mean4(sqrt(Vgroup));Jgap=sqrt(Vbar)-Abar;Pgap=sqrt(V120)-sqrt(Vbar);Rgap=Abar-A30_first;delta=Jgap+Pgap+Rgap",
                     ss_identity="119*var120=29*sum4(var30group)+30*sum4((mean_group-mean120)^2), separately for lower/upper",
                     interpretation="Q is repeated-measurement variance, not V/endpoint variance or original-level MCSE; Jgap nonnegative is algebra not scientific hypothesis; no ideal/infiniteK absolute bias/c4 correction/trueK120 claim",
                     t_control="t computed once per original and broadcast; allA/V/Q/gaps0, actual interval count one original not J*K; any missing required logical t slot invalidates that control; zero-SE comparison boundsNA; oldM05K1 staysNA",
                     summaries="original-level N10/seed,N30/population, J/K/groups never counted as originals; by_K fields averaged over J first, Q per original; mean/SD/MCSE/t99% intervals and linear p05/25/50/75/95, no clipping/trimming",
                     summary_aliases="A=mean_A30/mean_A120,V=mean_V30/mean_V120,Q=Q30/Q120; Ase/Vse aliases refer to same finite-K operational quantities; C/T/width/warnings purely descriptive",
                     ratios="mean_individual_Q120_Q30 separately from ratio_mean_Q120_over_mean_Q30; zero-denominator original ratios NA, conditional_ complete finite-ratio references explicit, never confirmation"),
        confirmation=dict(H1="per population mean(d),99%t df29; strict inside(-.0025,.0025)=>supported small movement; lo>.0025 or hi<-.0025=>not_supported; touching/crossing=>inconclusive",
                          H2="per population mean(W),99%t df29; hi<0=>supported >50% repeated-VARIANCE reduction,lo>=0=>not_supported; otherwise inconclusive",
                          joint="each hypothesis supports only if both populations support; any not_supported dominates overall even if other population incomplete, else inconclusive",
                          intervals=4,confidence=.99,seed_gate=False,units="Ase units, not percentage points; 50% concerns Q variance not endpoint SD",
                          completeness="all J20*K120 endpoints for that original/method required; invalid/missing/nonfinite/se<=0 or primarySE0=>mainNA/inconclusive; other method invalids do not poison completeBCa; conditional_ references never decide",
                          approximation="model-based original-level approximate t intervals for IID originals selected by fixed ID, finiteJ/K noise, N30 may be insufficiently normal; not inner-randomness-only intervals conditional on these60 samples; approximate union noncoverage sum~<=4%, no exact guarantee"),
        numeric_tolerances=dict(numeric_rtol=NUMERIC_RTOL,numeric_atol=NUMERIC_ATOL,identity_rtol=IDENTITY_RTOL,identity_atol=IDENTITY_ATOL,
                                identity_rule="abs(residual)<=identity_atol+identity_rtol*scale; endpoint scale=max(abs(lhs),abs(rhs)); gap Jgap>=-same tolerance; normalized residual divides bymax(1,scale)",
                                provenance="locked on deterministic fixtures before main; never relaxed to fit outcomes"),
        storage=dict(panel_axes=["original","replicate","method","inner"],diagnostics="float64 lower/upper/z0/a/q/tau; boolean valid/tail_warning; uint8 reason_codes/U96dictionary; samplesfloat64/hashS64/outeruint16/replicate&inneruint8; observed_streams bool(original,replicate,inner) distinguishes uncomputed from scientifically invalid slots",
                     derived_tables=dict(original_metrics=180,original_K_metrics=360,replicate_metrics_gz=3600,group_metrics_gz=14400,K_metrics_gz=7200,summary=36,pooled_summary=12,decomposition_summary=24),
                     selected_streams=[dict(distribution=d,master_seed=s,outer_id=0,replicate_id=0,inner_id=k) for d,s in fixed_cells() for k in (0,29,30,119)],
                     stream_schema="indicesuint8(4999,20) reversible from nativeint64 RNG,meansfloat64(4999),originalhash/IDs; exactly24 stored; other143976 generation sequences not reconstructable",
                     signs="raw Jgap/Pgap/Rgap/delta values and exact -1/0/+1 signs retained for every original/r/method, with algebra_ok and signed/normalized residuals; descriptive sign summaries do not count J as independent original N",
                     estimates=dict(raw_endpoint_numeric_bytes=27648000,raw_mask_reason_bytes=1296000,raw_original_bytes=9600,raw_selected_stream_bytes=3359328,
                                    panel_reserve_bytes=29*1024**2,stream_reserve_bytes=int(3.5*1024**2),csv_reserve_bytes=int(4.5*1024**2),single_html_reserve_bytes=5*1024**2,other_public_reserve_bytes=4*1024**2,total_public_reserve_bytes=46*1024**2),
                     estimate_caveat="conservative allocation not guaranteed compression; runtime guards stop without scientific resizing if actual limits reached"),
        policy=POLICY,execution=dict(main="once only, root uses guarded CLI; ordinary tests/UI/audit never launch main",smoke="sole separate namespace6savedouter0*J2*K8*B99,6 originals/96 streams/9504resamples,4groups*2; no scientific selection",
                                     stop="worker294s/parentTERM295/KILL298/finish299, externalCLIcap300; preserve partial panels/rows/proof/attempt, no resize/retry on outcome/invalid/timeout",
                                     correction="actual code bug only, at most one separately authorized attempt retaining oldsource/results/reason, cumulative mainwall<=600s"),
        environment=runtime_environment(),source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})


def validate_manifest(m):
    if m.get("schema_version")!=1 or m.get("question_id")!="RQ-M07" or m.get("design")!=DESIGN or m.get("policy")!=POLICY:raise ValueError("Manifest differs from fixed RQ-M07")


def verify_committed_inputs(path,m,commit):
    if len(commit)!=40 or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=commit:raise ValueError("--source-commit must be full current reviewed HEAD")
    for name in [*SOURCES,str(path.resolve().relative_to(ROOT))]:
        if subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=ROOT)!=(ROOT/name).read_bytes():raise ValueError(f"Committed source differs: {name}")
    for name in SOURCES:
        if sha256(ROOT/name)!=m["source_files_sha256"].get(name):raise ValueError(f"Preregistered source differs: {name}")
    if runtime_environment()!=m["environment"]:raise ValueError("Runtime environment differs")
    validate_inputs_manifest(m["inputs"],root=ROOT)


def enforce_budget(wall,rss,output):
    if wall>=294 or rss>=2*1024**3 or output+POLICY["public_overhead_reserve_bytes"]+65536>=50*1024**2:raise BudgetExceeded("wall/RSS/output budget reached, including reserved publication overhead")


def _begin_cleanup():
    signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGTERM,signal.SIG_IGN)


def require_output_capacity(output,pending_bytes):
    if directory_bytes(output)+pending_bytes+POLICY["public_overhead_reserve_bytes"]+65536>=POLICY["output_bytes"]:raise BudgetExceeded("Output capacity reserved before write")


def npz_size_bound(payload):
    # Raw array bytes plus per-array NPY/ZIP headers and conservative DEFLATE
    # overhead. Compression is never assumed when reserving an upcoming write.
    raw=sum(np.asarray(value).nbytes+1024 for value in payload.values())
    return int(raw*1.02)+65536


def _allocate(saved,R,J,K):
    shape=(R,J,3,K)
    return {**{key:np.full(shape,np.nan) for key in DIAGNOSTICS},"valid":np.zeros(shape,bool),"tail_warning":np.zeros(shape,bool),"reason_codes":np.zeros(shape,np.uint8),
            "observed_streams":np.zeros((R,J,K),bool),"samples":np.array(saved["samples"][:R],dtype=float,copy=True),"sample_sha256":np.array(saved["sample_sha256"][:R],dtype="S64"),"outer_id":np.array(saved["outer_id"][:R],dtype=np.uint16)}


def derived_tables(result,raw,se,dist,seed,outer,digest,R,J,K,B=4999):
    meta=dict(distribution=dist,master_seed=seed,outer_id=outer,n=20,R=R,J=J,K_total=K,B=B,alpha=.05,se=se,sample_sha256=digest)
    original,rep,by_k=result["original"],result["replicate"],result["by_K"]
    originals,original_K,reps,groups,K_rows=[],[],[],[],[]
    for mi,method in enumerate(METHODS):
        row={**meta,"method":method};row.update({key:value[mi].item() for key,value in original.items()});originals.append(row)
        for ki,limit in enumerate((K//4,K)):
            lo,hi=raw["lower"][:,mi,:limit],raw["upper"][:,mi,:limit]
            with np.errstate(over="ignore",invalid="ignore"):width=hi-lo
            mask=raw["valid"][:,mi,:limit]&np.isfinite(lo)&np.isfinite(hi)&(lo<=hi)&np.isfinite(width)
            count=int(original["valid_inner"][mi]) if method=="t" else int(mask.sum());total=1 if method=="t" else J*limit
            krow={**meta,"method":method,"K":limit,"complete":bool(original["complete"][mi]),"total_inner":total,"valid_inner":count,"invalid_inner":total-count}
            krow.update({key:value[ki,mi].item() for key,value in by_k.items()})
            krow.update(A=original["mean_A30" if ki==0 else "mean_A120"][mi].item(),V=original["mean_V30" if ki==0 else "mean_V120"][mi].item(),Q=original["Q30" if ki==0 else "Q120"][mi].item());original_K.append(krow)
    for r in range(J):
        for mi,method in enumerate(METHODS):
            rrow={**meta,"method":method,"replicate_id":r}
            rrow.update({key:value[r,mi].item() for key,value in rep.items() if key!="by_K" and key not in GROUP_FIELDS})
            for gap in GAPS:rrow[gap+"_sign"]=int(np.sign(rrow[gap])) if np.isfinite(rrow[gap]) else "NA"
            reps.append(rrow)
            for group in range(4):
                grow={**meta,"method":method,"replicate_id":r,"group":group,"group_start":group*(K//4),"group_stop":(group+1)*(K//4)}
                grow.update({key.removeprefix("group_"):rep[key][r,mi,group].item() for key in GROUP_FIELDS});grow["complete"]=all(np.isfinite(grow[key.removeprefix("group_")]) for key in GROUP_FIELDS);groups.append(grow)
            for ki,limit in enumerate((K//4,K)):
                lo,hi=raw["lower"][r,mi,:limit],raw["upper"][r,mi,:limit]
                with np.errstate(over="ignore",invalid="ignore"):width=hi-lo
                mask=raw["valid"][r,mi,:limit]&np.isfinite(lo)&np.isfinite(hi)&(lo<=hi)&np.isfinite(width)
                count=int(rep["valid_inner"][r,mi]) if method=="t" else int(mask.sum());total=1 if method=="t" else limit
                krow={**meta,"method":method,"replicate_id":r,"K":limit,"complete":bool(rep["complete"][r,mi]),"total_inner":total,"valid_inner":count,"invalid_inner":total-count}
                krow.update({key:value[r,ki,mi].item() for key,value in rep["by_K"].items()});K_rows.append(krow)
    return dict(originals=originals,original_K=original_K,replicates=reps,groups=groups,K_rows=K_rows)


def run_cell(dist,seed,saved,R=10,J=20,K=120,B=4999,namespace=MAIN_NAMESPACE,check=lambda:None,bootstrap_fn=None,audit_sink=None,progress=None):
    if not isinstance(R,int) or isinstance(R,bool) or not 1<=R<=10 or not isinstance(J,int) or isinstance(J,bool) or not 2<=J<=20 or not isinstance(K,int) or isinstance(K,bool) or not 8<=K<=120 or K%4 or not isinstance(B,int) or isinstance(B,bool) or not 1<=B<=4999:raise ValueError("Invalid fixed-panel fixture dimensions")
    if np.asarray(saved["samples"]).shape[1:]!=(20,) or len(saved["samples"])<R or np.asarray(saved["outer_id"][:R]).tolist()!=list(range(R)):raise ValueError("Saved input selection shape/IDs differ")
    data=_allocate(saved,R,J,K);labels=[""];tables={key:[] for key in ("originals","original_K","replicates","groups","K_rows")};bootstrap_fn=bootstrap_fn or generate_bootstrap_means
    progress=progress if progress is not None else {};progress.update(panel=data,reason_labels=labels,observed_original_count=0,retained_original_count=0,observed_stream_count=0,tables=tables)
    for oi,outer in enumerate(data["outer_id"]):
        progress.update(active_original_id=int(outer),active_original_index=oi,retained_original_count=oi+1)
        check();sample=data["samples"][oi];digest=sample_hash(sample)
        if digest!=data["sample_sha256"][oi].decode("ascii"):raise ValueError("Saved original hash mismatch")
        context=prepare_sample(sample,alpha=.05);se=float(context["se"])
        for r in range(J):
            for inner in range(K):
                check();failure="";audit=namespace==MAIN_NAMESPACE and int(outer)==0 and r==0 and inner in (0,29,30,119) and audit_sink is not None
                if not context["valid"]:means=np.full(B,np.nan)
                else:
                    try:
                        generated=bootstrap_fn(sample,dist,n=20,master_seed=seed,outer_id=int(outer),replicate_id=r,inner_id=inner,Bmax=B,namespace=namespace,max_elements=1000000,return_indices=audit)
                        if audit:
                            means,indices=generated;audit_sink(dict(indices=indices,means=means,distribution=dist,master_seed=seed,outer_id=int(outer),replicate_id=r,inner_id=inner,n=20,B=B,J=J,K=K,alpha=.05,namespace=namespace,native_indices_dtype="int64",stored_indices_dtype="uint8",sample_sha256=digest.encode("ascii")))
                        else:means=generated
                    except ValueError as exc:
                        if str(exc)!="bootstrap_mean_nonfinite":raise
                        means=np.full(B,np.nan);failure=str(exc)
                records=infer_single(context,means)
                for mi,method in enumerate(METHODS):
                    record=records[method]
                    for field in DIAGNOSTICS:data[field][oi,r,mi,inner]=record[field]
                    data["valid"][oi,r,mi,inner]=record["valid"];data["tail_warning"][oi,r,mi,inner]=record["tail_warning"]
                    reason=failure if failure and method!="t" else record["reason"]
                    if reason not in labels:
                        if len(labels)>=256 or len(reason)>96:raise ValueError("Reason dictionary capacity exceeded")
                        labels.append(reason)
                    data["reason_codes"][oi,r,mi,inner]=labels.index(reason)
                data["observed_streams"][oi,r,inner]=True;progress["observed_stream_count"]+=1
        raw={key:value[oi] for key,value in data.items() if key in (*DIAGNOSTICS,"valid","tail_warning")}
        result=original_metrics(raw,se,group_size=K//4,expected_replicates=J)
        local=derived_tables(result,raw,se,dist,seed,int(outer),digest,R,J,K,B=B)
        for key in tables:tables[key].extend(local[key])
        progress["observed_original_count"]=oi+1
    check();data["reason_labels"]=np.array(labels,dtype="U96")
    return data,tables


def _index(rows,by_K=False):
    result={}
    for row in rows:
        key=(row["distribution"],row["master_seed"],row["method"])+( (row["K"],) if by_K else () )
        group=result.setdefault(key,{})
        if row["outer_id"] in group or row["outer_id"] not in range(10):raise ValueError("Duplicate/out-of-range original ID")
        group[row["outer_id"]]=row
    return result


def _expected(index,dist,seed,method,field,K=None):
    key=(dist,seed,method)+((K,) if K is not None else ());group=index.get(key,{})
    return np.array([group[i][field] if i in group and group[i]["complete"] else np.nan for i in range(10)])


def _values(index,dist,seeds,method,field,K=None):
    return np.concatenate([_expected(index,dist,s,method,field,K) for s in seeds])


def metric_statistics(values):
    values=np.asarray(values,dtype=float);stats=mean_t_interval(values,confidence=.99)
    quantiles=np.quantile(values,QUANTILES,method="linear") if len(values) and np.isfinite(values).all() else np.full(5,np.nan)
    return dict(mean=stats["mean"],sd=stats["sd"],mcse=stats["mcse"],decision_low=stats["low"],decision_high=stats["high"],**{f"p{int(q*100):02d}":float(value) for q,value in zip(QUANTILES,quantiles)})


def summarize(original_K,pooled=False):
    index=_index(original_K,by_K=True);output=[]
    for dist in DISTRIBUTIONS:
        for seed in (["pooled"] if pooled else (0,1,2)):
            seeds=(0,1,2) if pooled else (seed,);planned=30 if pooled else 10
            for method in METHODS:
                for K in (30,120):
                    group=[row for s in seeds for row in index.get((dist,s,method,K),{}).values()];complete=[r for r in group if r["complete"]]
                    row=dict(distribution=dist,master_seed=seed,n=20,R=planned,J=20,K=K,K_total=120,B=4999,alpha=.05,method=method,original_total=planned,observed_original_count=len(group),missing_original_count=planned-len(group),
                             complete_original_count=len(complete),incomplete_original_count=planned-len(complete),interval_count=sum(r["total_inner"] for r in group),valid_count=sum(r["valid_inner"] for r in group),invalid_count=sum(r["invalid_inner"] for r in group),conditional_original_count=len(complete))
                    for field in SUMMARY_METRICS:
                        values=_values(index,dist,seeds,method,field,K)
                        for prefix,array in (("",values),("conditional_",values[np.isfinite(values)])):
                            for suffix,value in metric_statistics(array).items():row[f"{prefix}{field}"+("" if suffix=="mean" else "_"+suffix)]=value
                    output.append(row)
    return output


def assemble_hypotheses(originals):
    index=_index(originals);quantities={};seed_details=[]
    for dist in DISTRIBUTIONS:
        for H,field,decision in (("H1","d",band_status),("H2","W",reduction_status)):
            values=_values(index,dist,(0,1,2),"bca",field);stats=mean_t_interval(values,confidence=.99)
            quantities[f"{H}_{dist}"]={**stats,**decision(stats["low"],stats["high"],complete=bool(stats["valid"])),"field":field,"population":dist,
                                       "conditional_interval":mean_t_interval(values[np.isfinite(values)],confidence=.99)}
    for seed in (0,1,2):
        seed_details.append(dict(master_seed=seed,**{f"{H}_{dist}":mean_t_interval(_expected(index,dist,seed,"bca",field),confidence=.99) for dist in DISTRIBUTIONS for H,field in (("H1","d"),("H2","W"))}))
    return dict(question_id="RQ-M07",confirmatory_intervals=4,decision_confidence=.99,independent_originals_per_population=30,target_method="bca",B=4999,J=20,K_first=30,K_full=120,seed_direction_gate=False,
                H1_margin=.0025,H1_units="Ase units, not percentage points",H2_variance_multiplier=.5,
                quantities=quantities,seed_details=seed_details,hypotheses={H:dict(status=joint_status([quantities[f"{H}_{d}"]["status"] for d in DISTRIBUTIONS]),**{d:quantities[f"{H}_{d}"]["status"] for d in DISTRIBUTIONS}) for H in ("H1","H2")})


def decomposition_summary(originals,replicates):
    index=_index(originals);output=[]
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2,"pooled"):
            seeds=(0,1,2) if seed=="pooled" else (seed,)
            for method in METHODS:
                group=[r for r in replicates if r["distribution"]==dist and r["method"]==method and r["master_seed"] in seeds]
                row=dict(distribution=dist,master_seed=seed,method=method,original_total=30 if seed=="pooled" else 10,diagnostic_replicates=len(group),interpretation="r-level signs/quantiles are descriptive; original-level intervals use N10/30")
                for gap in GAPS:
                    raw=np.array([r[gap] for r in group],dtype=float);finite=raw[np.isfinite(raw)]
                    row.update({gap+"_finite_count":len(finite),gap+"_negative_count":int((finite<0).sum()),gap+"_zero_count":int((finite==0).sum()),gap+"_positive_count":int((finite>0).sum()),gap+"_raw_mean":float(finite.mean()) if len(finite) else np.nan})
                    for q,value in zip(QUANTILES,np.quantile(finite,QUANTILES,method="linear") if len(finite) else np.full(5,np.nan)):row[f"{gap}_raw_p{int(q*100):02d}"]=float(value)
                    values=_values(index,dist,seeds,method,"mean_"+gap)
                    for prefix,array in (("",values),("conditional_",values[np.isfinite(values)])):
                        for key,value in mean_t_interval(array,confidence=.99).items():row[f"{prefix}{gap}_original_{key}"]=value
                for field in RESIDUALS:
                    values=np.array([r[field] for r in group],dtype=float);finite=values[np.isfinite(values)]
                    row[field+"_max_abs"]=float(np.max(np.abs(finite))) if len(finite) else np.nan
                row["algebra_failed_replicates"]=sum(r["reason"]=="algebraic_identity_failed" for r in group)
                row["algebra_unchecked_replicates"]=sum(not r["algebra_ok"] and r["reason"]!="algebraic_identity_failed" for r in group)
                row["algebra_passed_replicates"]=sum(bool(r["algebra_ok"]) for r in group)
                ratio=_values(index,dist,seeds,method,"ratio_Q120_Q30")
                q30=_values(index,dist,seeds,method,"Q30");q120=_values(index,dist,seeds,method,"Q120")
                finite=np.isfinite(ratio);paired=np.isfinite(q30)&np.isfinite(q120)
                row.update(ratio_planned_originals=len(ratio),ratio_finite_originals=int(finite.sum()),ratio_missing_or_undefined_originals=int((~finite).sum()),
                           mean_individual_Q120_Q30=float(ratio.mean()) if finite.all() else np.nan,
                           conditional_mean_individual_Q120_Q30=float(ratio[finite].mean()) if finite.any() else np.nan,
                           ratio_mean_Q120_over_mean_Q30=float(q120.mean()/q30.mean()) if paired.all() and q30.mean()>0 else np.nan,
                           conditional_ratio_mean_Q120_over_mean_Q30=float(q120[paired].mean()/q30[paired].mean()) if paired.any() and q30[paired].mean()>0 else np.nan,
                           conditional_ratio_of_means_originals=int(paired.sum()),ratio_reference="descriptive only; mean of individual ratios differs from ratio of population means; zero denominator NA")
                output.append(row)
    return output


class TableWriter:
    def __init__(self,handle):self.handle=handle;self.writer=None
    def write(self,row):
        if self.writer is None:self.writer=csv.DictWriter(self.handle,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row)
    def write_rows(self,rows):
        for row in rows:self.write(row)


@contextmanager
def gzip_table(path):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="",mode="wb",compresslevel=6,mtime=0,fileobj=raw) as zipped:
            with io.TextIOWrapper(zipped,encoding="utf-8",newline="") as text:yield TableWriter(text)


def save_panel(path,panel,dist,seed,J,K,B,planned=10,capacity_output=None):
    payload=dict(**panel,methods=np.array(METHODS),replicate_id=np.arange(J,dtype=np.uint8),inner_id=np.arange(K,dtype=np.uint8),distribution=dist,master_seed=seed,n=20,R=planned,observed_original_count=len(panel["outer_id"]),completed_original_count=int(np.all(panel["observed_streams"],axis=(1,2)).sum()),observed_stream_count=int(panel["observed_streams"].sum()),J=J,K=K,K_first=K//4,B=B,alpha=.05,namespace=MAIN_NAMESPACE,input_accepted_head=ACCEPTED_HEAD)
    if capacity_output is not None:require_output_capacity(capacity_output,npz_size_bound(payload))
    with path.open("xb") as handle:np.savez_compressed(handle,**payload)


def write_figure(pooled,hypotheses,output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    output.mkdir(exist_ok=False);fig=make_subplots(rows=1,cols=2,subplot_titles=["BCa mean A at K30/120 (original-level99% bars)","BCa repeated-measurement variance Q"])
    for dist in DISTRIBUTIONS:
        rows=[next(r for r in pooled if r["distribution"]==dist and r["method"]=="bca" and r["K"]==K) for K in (30,120)]
        for col,field in ((1,"A"),(2,"Q")):
            fig.add_trace(go.Scatter(x=["30 first","120"],y=[r[field] for r in rows],mode="lines+markers",name=dist,legendgroup=dist,showlegend=col==1,
                                    error_y=dict(type="data",symmetric=False,array=[r[field+"_decision_high"]-r[field] for r in rows],arrayminus=[r[field]-r[field+"_decision_low"] for r in rows])),row=1,col=col)
    fig.update_layout(template="plotly_white",width=1250,height=600,title="RQ-M07: BCa4conditions; saved originals N30/population, J20, B4999; K120 is finite, not truth")
    fig.write_html(str(output/"overview.html"),include_plotlyjs=True,full_html=True,config={"displaylogo":False})


def _final_resources(output,record):
    record.update(process_wall_seconds=perf_counter()-_PROCESS_START,process_peak_rss_bytes=peak_rss_bytes(),output_bytes_before_resources=directory_bytes(output));record["output_bytes_total"]=record["output_bytes_before_resources"]
    for _ in range(5):
        total=record["output_bytes_before_resources"]+len((json.dumps(record,indent=2,allow_nan=False)+"\n").encode())
        if total==record["output_bytes_total"]:break
        record["output_bytes_total"]=total
    _write_json(output/"resources.json",record)


def run_benchmark(manifest_path,output,source_commit):
    m=json.loads(manifest_path.read_text());validate_manifest(m);verify_committed_inputs(manifest_path,m,source_commit);inputs=load_inputs(m["inputs"],root=ROOT)
    output=output.resolve()
    if ROOT not in output.parents:raise ValueError("Output must be inside project")
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ("attempt.json","resources.json","panels","streams","original_metrics.csv","summary.csv","figures")):raise FileExistsError("Existing attempt/results preserved")
    attempt=dict(question_id="RQ-M07",status="started",started_utc=datetime.now(timezone.utc).isoformat(),source_git_sha=source_commit,manifest_sha256=sha256(manifest_path),namespace=MAIN_NAMESPACE,single_main_attempt=True)
    _write_json(output/"attempt.json",attempt);(output/"panels").mkdir();(output/"streams").mkdir()
    originals,original_K,replicates,completed=[],[],[],[]
    observed_streams,observed_invalid=0,0
    resources={**attempt,"environment":runtime_environment(),"R":10,"J":20,"K":120,"K_first":30,"B":4999,"n":20,"total_cells":6,"mother_samples_generated":0,"planned_audit_streams":24,"all_other_bootstrap_arrays_saved":False,"max_resample_elements":1000000,
               "main_attempts":1,"reruns":0,"planned_originals":60,"planned_inner_streams":144000,"planned_logical_intervals":432000,
               "wall_scope":"module start before heavy imports through outputs, excluding interpreter startup/final resource write; outer guard/time measure full CLI"}
    budget_state=dict(calls=0,size=directory_bytes(output))
    def check(force=False):
        budget_state["calls"]+=1
        if force or budget_state["calls"]%128==1:budget_state["size"]=directory_bytes(output)
        enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),budget_state["size"])
    def audit_sink(payload):
        path=output/"streams"/f"{payload['distribution']}_seed{payload['master_seed']}_outer{payload['outer_id']}_replicate{payload['replicate_id']}_inner{payload['inner_id']}.npz"
        require_output_capacity(output,npz_size_bound(payload)+POLICY["cell_cleanup_reserve_bytes"])
        with path.open("xb") as handle:np.savez_compressed(handle,**payload)
        check(force=True)
    try:
        with gzip_table(output/"replicate_metrics.csv.gz") as rep_writer,gzip_table(output/"group_metrics.csv.gz") as group_writer,gzip_table(output/"K_metrics.csv.gz") as K_writer:
            for dist,seed in fixed_cells():
                resources["active_cell"]=[dist,seed];progress={}
                require_output_capacity(output,POLICY["cell_cleanup_reserve_bytes"])
                try:panel,tables=run_cell(dist,seed,inputs[(dist,seed)],check=check,audit_sink=audit_sink,progress=progress)
                except Exception:
                    _begin_cleanup()
                    observed=progress.get("observed_original_count",0);retained=progress.get("retained_original_count",0)
                    resources.update(partial_completed_original_count=observed,partial_retained_original_count=retained,partial_observed_stream_count=progress.get("observed_stream_count",0),partial_active_original_id=progress.get("active_original_id"))
                    observed_streams+=progress.get("observed_stream_count",0)
                    if retained:
                        mask=progress["panel"]["observed_streams"][:,:,None,:]
                        observed_invalid+=int((~progress["panel"]["valid"]&mask).sum())
                    if retained:
                        partial={key:value[:retained] for key,value in progress["panel"].items()};partial["reason_labels"]=np.array(progress["reason_labels"],dtype="U96")
                        save_panel(output/"panels"/f"{dist}_seed{seed}_partial.npz",partial,dist,seed,20,120,4999,capacity_output=output)
                        local=progress["tables"];originals.extend(local["originals"]);original_K.extend(local["original_K"]);replicates.extend(local["replicates"])
                        rep_writer.write_rows(local["replicates"]);group_writer.write_rows(local["groups"]);K_writer.write_rows(local["K_rows"])
                    raise
                save_panel(output/"panels"/f"{dist}_seed{seed}.npz",panel,dist,seed,20,120,4999,capacity_output=output)
                observed_streams+=int(panel["observed_streams"].sum());observed_invalid+=int((~panel["valid"]).sum())
                originals.extend(tables["originals"]);original_K.extend(tables["original_K"]);replicates.extend(tables["replicates"])
                rep_writer.write_rows(tables["replicates"]);group_writer.write_rows(tables["groups"]);K_writer.write_rows(tables["K_rows"]);completed.append([dist,seed]);check(force=True)
        require_output_capacity(output,int(4.5*1024**2)+5*1024**2)
        write_csv(output/"original_metrics.csv",originals);write_csv(output/"original_K_metrics.csv",original_K)
        pooled=summarize(original_K,pooled=True);write_csv(output/"summary.csv",summarize(original_K));write_csv(output/"pooled_summary.csv",pooled)
        write_csv(output/"decomposition_summary.csv",decomposition_summary(originals,replicates))
        hypotheses=assemble_hypotheses(originals);_write_json(output/"hypotheses.json",hypotheses);write_figure(pooled,hypotheses,output/"figures")
        check(force=True);resources["status"]="completed"
    except Exception as exc:
        _begin_cleanup()
        resources.update(status="stopped",stop_reason=type(exc).__name__,stop_detail=str(exc) if isinstance(exc,BudgetExceeded) else "See preserved diagnostics; no automatic retry")
        if originals and not (output/"original_metrics.csv").exists():write_csv(output/"original_metrics.csv",originals)
        if original_K and not (output/"original_K_metrics.csv").exists():write_csv(output/"original_K_metrics.csv",original_K)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGTERM,signal.SIG_IGN)
        resources.update(completed_cells=len(completed),completed_cell_keys=completed,incomplete_cells=[list(c) for c in fixed_cells() if list(c) not in completed],
                         completed_originals=len(originals)//3,observed_inner_streams=observed_streams,observed_logical_intervals=3*observed_streams,observed_invalid_logical_intervals=observed_invalid,
                         bootstrap_samples=observed_streams*4999,logical_resampling_indices=observed_streams*4999*20,
                         saved_audit_streams=len(list((output/"streams").glob("*.npz"))),finished_utc=datetime.now(timezone.utc).isoformat());_final_resources(output,resources)
    return resources


def run_smoke(output):
    provenance=make_inputs_manifest(root=ROOT);inputs=load_inputs(provenance,root=ROOT);output=output.resolve()
    if ROOT not in output.parents:raise ValueError("Smoke output must be inside project")
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError("Existing smoke attempt preserved")
    attempt=dict(question_id="RQ-M07",scope="smoke only; savedouter0/J2/K8/B99, no scientific decisions",R=1,J=2,K=8,B=99,namespace=SMOKE_NAMESPACE,source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})
    _write_json(output/"attempt.json",attempt);result,completed={**attempt,"environment":runtime_environment(),"status":"started"},[]
    try:
        for dist,seed in fixed_cells():
            panel,tables=run_cell(dist,seed,inputs[(dist,seed)],R=1,J=2,K=8,B=99,namespace=SMOKE_NAMESPACE,check=lambda:enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output)))
            completed.append(dict(distribution=dist,master_seed=seed,valid_entries=int(panel["valid"].sum())))
        result["status"]="completed"
    except Exception as exc:result.update(status="stopped",stop_reason=type(exc).__name__);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGTERM,signal.SIG_IGN)
        result.update(completed_cells=len(completed),cells=completed,mother_samples_generated=0,saved_originals_used=len(completed),inner_streams=16*len(completed),bootstrap_samples=1584*len(completed),logical_resampling_indices=31680*len(completed));_final_resources(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-manifest",action="store_true");modes.add_argument("--run",action="store_true");modes.add_argument("--smoke",action="store_true")
    parser.add_argument("--manifest",type=Path,default=ROOT/"evidence/rq_m07/manifest.json");parser.add_argument("--output",type=Path);parser.add_argument("--source-commit")
    args=parser.parse_args()
    if args.prepare_manifest:
        path=args.manifest.resolve()
        if ROOT not in path.parents:parser.error("Manifest must be inside project")
        path.parent.mkdir(parents=True,exist_ok=True);_write_json(path,make_manifest());print("Prepared",path.relative_to(ROOT))
    elif args.smoke:
        result=run_smoke(args.output or ROOT/"evidence/rq_m07/smoke");print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))
    else:
        if not args.source_commit:parser.error("--run requires full reviewed --source-commit")
        result=run_benchmark(args.manifest.resolve(),args.output or ROOT/"evidence/rq_m07",args.source_commit);print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))


if __name__=="__main__":main()
