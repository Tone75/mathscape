"""Preregister/run RQ-M06 using accepted saved originals only, never generation.

The guarded CLI executes main once. Ordinary tests and UI never start main.
The sole smoke uses saved outer0,1, K2 and B99 in its separate namespace.
"""

from time import perf_counter
_PROCESS_START=perf_counter()

import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
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

from mathscape.matched_stability import (
    make_inputs_manifest,validate_inputs_manifest,load_inputs,generate_bootstrap_means,
    prepare_sample,infer_prefixes,original_metrics,mean_t_interval,welch_interval,
    practical_difference_status,reduction_status,combine_reduction_status,
)
from scripts.run_bootstrap_benchmark import environment,sha256,peak_rss_bytes,directory_bytes,_write_json,write_csv
from scripts.run_stability_benchmark import sample_hash

DISTRIBUTIONS=("gamma","lognormal")
METHODS=("t","percentile","bca")
MAIN_NAMESPACE,SMOKE_NAMESPACE=0x4D3036,0x534D3036
ACCEPTED_HEAD="dc2501c3099f5c25227d9eee00d227c2f0b02f81"
DESIGN=dict(distributions=list(DISTRIBUTIONS),n=20,master_seeds=[0,1,2],R=200,K=30,B_values=[999,4999],alpha=.05,
            outer_ids=list(range(200)),true_mean=0.,true_variance=1.,true_skewness=2.,methods=list(METHODS),source_question="RQ-M05",input_accepted_head=ACCEPTED_HEAD)
POLICY=dict(wall_seconds=300,internal_stop_seconds=294,cleanup_reserve_seconds=6,peak_rss_bytes=2*1024**3,
            output_bytes=50*1024**2,max_resample_elements=1000000,decision_confidence=.99,confirmatory_intervals=3,
            practical_Ase_margin=.01,reduction_reference_multiplier=.75)
SOURCES=("mathscape/matched_stability.py","mathscape/matched_moments.py","mathscape/stability.py","mathscape/bootstrap.py","mathscape/inference.py",
         "scripts/run_matched_stability_benchmark.py","scripts/run_matched_stability_guarded.py","scripts/audit_matched_stability.py",
         "scripts/run_bootstrap_benchmark.py","scripts/run_stability_benchmark.py",
         "tests/test_matched_stability.py","tests/test_matched_stability_runner.py","tests/test_matched_stability_guard.py","tests/test_matched_stability_audit.py",
         "requirements.txt","evidence/rq_m06/SPEC.md")
DIAGNOSTICS=("lower","upper","z0","a","q_low","q_high","tau_low","tau_high")
METRICS=("lower_sd","upper_sd","S","Ase","Apop","Vse","C","Mbelow","Mabove","T","mean_width","warning_rate")
METHOD_PAIRS=(("percentile","t"),("bca","percentile"),("bca","t"))
QUANTILES=(.05,.25,.50,.75,.95)


def runtime_environment():
    return {**environment(),"streamlit":version("streamlit"),"pytest":version("pytest")}


def fixed_cells():
    return [(dist,seed) for dist in DISTRIBUTIONS for seed in (0,1,2)]


def make_manifest():
    inputs=make_inputs_manifest(root=ROOT)
    return dict(
        schema_version=1,question_id="RQ-M06",created_utc=datetime.now(timezone.utc).isoformat(),design=DESIGN,inputs=inputs,
        totals=dict(input_cells=6,original_samples=1200,inner_streams=36000,bootstrap_samples=179964000,logical_resampling_indices=3599280000,
                    logical_intervals=216000,original_metric_rows=7200,seed_summary_rows=36,pooled_summary_rows=12),
        input_contract="accepted M05 saved originals only, outer0..199 fixed before inspecting endpoints/warnings/sample skewness; accepted git blobs/local file hashes/all selected hashes and sorted IDs verified; no mother-sample generation or sample standardization",
        reuse_disclosure="designers know M05 aggregate results; this is reuse of an accepted saved panel, not an unused independent replication; no mixing of M05 inner0 or updating/pooling M05 C/T conclusions",
        randomness=dict(generator="Generator(PCG64(SeedSequence(...)))",entropy="existing master_seed",main_namespace=MAIN_NAMESPACE,smoke_namespace=SMOKE_NAMESPACE,
                        distribution_ids=dict(gamma=1,lognormal=2),spawn_key=["namespace","distribution_id","n","outer_id","purpose_id","inner_id"],
                        purpose_ids=dict(bootstrap=1),original_generation="purpose0 never called",inner_ids=list(range(30)),outer_ids=list(range(200)),
                        excluded_from_key=["B","method","UI_state","batch_size"],native_indices_dtype="int64",
                        prefix="one length4999 mean vector per saved original/inner; first999 shared; percentile and BCa share means; recompute each prefix z0/q; prepare t/a/se once per original",
                        independence="G/L are independent original groups; only B/method differences within the same original are paired; K/B are not population sample counts"),
        inference="accepted ddof1 t, linear quantiles, half-tie z0, delete-one jackknife acceleration and BCa validity rules unchanged; no truth/population parameters passed to inference; no fallback/clipping/adaptive B; tau<10 warns without exclusion",
        metrics=dict(lower_sd="within-original SD of K30 lower endpoints,ddof1",upper_sd="within-original SD of K30 upper endpoints,ddof1",
                     S="hypot(lower_sd,upper_sd)/sqrt(2)",Ase="S/(sampleSD_ddof1/sqrt(20))",Apop="S/(1/sqrt(20)), evaluation only",Vse="Ase squared, descriptive",
                     interpretation="finiteK30 operational endpoint-computation variability, not bootstrap-mean SD/estimator SE/RMSE/unbiased ideal conditional SD; never pool endpoints across originals to compute SD",
                     t_control="t broadcast over B/K is deterministic: endpoint SD/S/Ase/Apop/Vse0, actual valid interval count is R not R*K; M05 K1 stability stays NA",
                     coverage="C=mean_k I(lower<=0<=upper),Mbelow=mean_k I(upper<0),Mabove=mean_k I(lower>0),T=Mbelow-Mabove; descriptive reused-subpanel only, no M05 C/T retest",
                     summaries="within-original quantities first; R200 per seed/N600 per population; mean/SD/MCSE/99% original-level t intervals, no clipping, zero variance bounds undefined",
                     quantiles="p05,p25,p50,p75,p95 linear over independent original metrics, for every recorded metric; no trimming/winsorization",
                     ratio="mean(Ase4999)/mean(Ase999), never mean of individual ratios"),
        confirmation=dict(
            H1="BCa B4999 meanAse(lognormal)-meanAse(gamma), independent Welch99%, N600/group. lo>.01 or hi<-.01 =>practical-difference supported; strict inside(-.01,.01)=>not_supported+equivalence_supported; otherwise touching/straddling=>inconclusive. .01 is sample-SE-normalized endpoint variability, NOT1pp",
            H2="per population D=Ase4999-.75*Ase999 on same original, mean t99%,N600. upper<0=>supported; lower>=0=>not_supported; otherwise inconclusive; overall supported iff both supported, not_supported if either not_supported, else inconclusive",
            intervals=3,confidence=.99,seed_gate=False,
            interpretation="model-based approximate original-level intervals for IID originals selected by fixed ID, including finiteK30 noise; not intervals conditional solely on these1200 fixed samples and inner randomness; approximate Bonferroni noncoverage sum~<=3%, no exact finite-sample guarantee",
            limits="B4999 is not true/infiniteB endpoint; stability reduction does not prove improved coverage/unbiasedness; limited averageAse equivalence does not guarantee individual stability or isolate kurtosis/tail/support cause"),
        invalid_policy="any required invalid K interval or se<=0 makes that method/B original primary metrics NA; required target missing/invalid=>main NA/inconclusive with planned denominators retained; unrelated methods/B do not poison complete targets; conditional_ references with explicit finite counts never decide",
        storage=dict(panel_axes=["original","B","method","inner"],samples="all1200 originalfloat64, sample_sha256S64 ASCII hash of little-endian float64 C-order bytes, outer_iduint16, inner_iduint8",
                     endpoints="float64 numeric diagnostics, boolean valid/tail_warning, uint8reason_codes indexing U96reason_labels; full t broadcast stored with R-only aggregation",
                     selected_streams=[dict(distribution=d,master_seed=s,outer_id=0,inner_id=k) for d,s in fixed_cells() for k in (0,1,29)],
                     stream_schema="indicesuint8(4999,20) reversible from nativeint64 RNG indices, meansfloat64(4999), inputhash/IDs/meta; first999 prefix; only18 preregistered streams stored",
                     audit_scope="all endpoint-derived metrics/tables/decisions plus exact indices->means/prefix/rank/a/linear endpoints for18 selected streams; never claim complete reconstruction of all36000 generation sequences",
                     estimate=dict(raw_original_bytes=192000,raw_endpoint_numeric_bytes=13824000,raw_mask_reason_bytes=648000,raw_hash_bytes=76800,
                                   raw_selected_stream_bytes=2519496,panel_reserve_bytes=18*1024**2,audit_stream_reserve_bytes=3*1024**2,
                                   csv_tables_reserve_bytes=6*1024**2,single_html_reserve_bytes=5*1024**2,other_public_reserve_bytes=8*1024**2,total_public_reserve_bytes=40*1024**2),
                     estimate_caveat="conservative allocation not guaranteed compression; actual guards stop without scientific resizing; no storage of other full bootstrap/index arrays"),
        policy=POLICY,execution=dict(main="once only; root executes guarded CLI, no ordinary pytest/UI/audit launch",smoke="sole separately authorized6cells/savedouter0,1/K2/B99 in separate namespace;12savedoriginals,24streams,2376resamples, no mother generation or scientific selection",
                                     guard="worker294softstop,parentTERM295/KILL298/finish target299, full external CLI cap300; cleanup ignores repeated TERM",
                                     correction="actual code bug only, at most one separately authorized attempt retaining reason/oldsource/results, cumulative mainwall<=600; no redraw on outcome/invalid/timeout",
                                     stop="preserve complete and observed partial panels/records/stream diagnostics; no automatic shrinking or ID/R/K/B changes"),
        environment=runtime_environment(),source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})


def validate_manifest(m):
    if m.get("schema_version")!=1 or m.get("question_id")!="RQ-M06" or m.get("design")!=DESIGN or m.get("policy")!=POLICY:raise ValueError("Manifest differs from fixed RQ-M06")


def verify_committed_inputs(path,m,commit):
    if len(commit)!=40 or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=commit:raise ValueError("--source-commit must be full current reviewed HEAD")
    for name in [*SOURCES,str(path.resolve().relative_to(ROOT))]:
        if subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=ROOT)!=(ROOT/name).read_bytes():raise ValueError(f"Committed source differs: {name}")
    for name in SOURCES:
        if sha256(ROOT/name)!=m["source_files_sha256"].get(name):raise ValueError(f"Preregistered source differs: {name}")
    if runtime_environment()!=m["environment"]:raise ValueError("Runtime environment differs")
    validate_inputs_manifest(m["inputs"],root=ROOT)


def enforce_budget(wall,rss,output):
    if wall>=294 or rss>=2*1024**3 or output+65536>=50*1024**2:raise BudgetExceeded("wall/RSS/output limit reached")


def _allocate(saved,R,K,Bs):
    shape=(R,len(Bs),3,K)
    return {**{field:np.full(shape,np.nan) for field in DIAGNOSTICS},"valid":np.zeros(shape,bool),"tail_warning":np.zeros(shape,bool),
            "reason_codes":np.zeros(shape,np.uint8),"samples":np.array(saved["samples"][:R],dtype=float,copy=True),
            "sample_sha256":np.array(saved["sample_sha256"][:R],dtype="S64"),"outer_id":np.array(saved["outer_id"][:R],dtype=np.uint16)}


def run_cell(distribution,seed,saved,R=200,K=30,Bs=(999,4999),namespace=MAIN_NAMESPACE,check=lambda:None,bootstrap_fn=None,audit_sink=None,progress=None):
    if not isinstance(R,int) or isinstance(R,bool) or not 1<=R<=200 or not isinstance(K,int) or isinstance(K,bool) or not 2<=K<=30:raise ValueError("R/K must be integer1..200/2..30")
    if not Bs or len(set(Bs))!=len(Bs) or any(not isinstance(b,int) or isinstance(b,bool) or not 1<=b<=4999 for b in Bs):raise ValueError("Invalid B prefix values")
    if np.asarray(saved["samples"]).shape[1:]!=(20,) or len(saved["samples"])<R or np.asarray(saved["outer_id"][:R]).tolist()!=list(range(R)):raise ValueError("Saved input IDs/shape differ from selection")
    bootstrap_fn=bootstrap_fn or generate_bootstrap_means;data=_allocate(saved,R,K,Bs);labels=[""];rows=[]
    progress=progress if progress is not None else {};progress.update(panel=data,reason_labels=labels,observed_original_count=0,rows=rows)
    for i,outer in enumerate(data["outer_id"]):
        check();sample=data["samples"][i];digest=sample_hash(sample)
        if digest!=data["sample_sha256"][i].decode("ascii"):raise ValueError("Selected saved original hash mismatch")
        context=prepare_sample(sample,alpha=.05);se=float(context["se"])
        for inner in range(K):
            check();failure="";audit=namespace==MAIN_NAMESPACE and int(outer)==0 and inner in (0,1,29) and audit_sink is not None
            if not context["valid"]:means=np.full(max(Bs),np.nan)
            else:
                try:
                    generated=bootstrap_fn(sample,distribution,n=20,master_seed=seed,outer_id=int(outer),inner_id=inner,Bmax=max(Bs),namespace=namespace,max_elements=1000000,return_indices=audit)
                    if audit:
                        means,indices=generated
                        audit_sink(dict(indices=indices,means=means,distribution=distribution,master_seed=seed,outer_id=int(outer),inner_id=inner,n=20,B=max(Bs),B_values=np.array(Bs),K=K,alpha=.05,namespace=namespace,
                                        native_indices_dtype="int64",stored_indices_dtype="uint8",sample_sha256=digest.encode("ascii")))
                    else:means=generated
                except ValueError as exc:
                    if str(exc)!="bootstrap_mean_nonfinite":raise
                    means=np.full(max(Bs),np.nan);failure=str(exc)
            records=infer_prefixes(context,means,B_values=Bs)
            for bi,B in enumerate(Bs):
                for mi,method in enumerate(METHODS):
                    record=records[B][method]
                    for field in DIAGNOSTICS:data[field][i,bi,mi,inner]=record[field]
                    data["valid"][i,bi,mi,inner]=record["valid"];data["tail_warning"][i,bi,mi,inner]=record["tail_warning"]
                    reason=failure if failure and method!="t" else record["reason"]
                    if reason not in labels:
                        if len(labels)>=256 or len(reason)>96:raise ValueError("Reason dictionary capacity exceeded")
                        labels.append(reason)
                    data["reason_codes"][i,bi,mi,inner]=labels.index(reason)
        metrics=original_metrics({key:array[i] for key,array in data.items() if key in (*DIAGNOSTICS,"valid","tail_warning")},se,n=20)
        for bi,B in enumerate(Bs):
            for mi,method in enumerate(METHODS):
                row=dict(distribution=distribution,master_seed=seed,outer_id=int(outer),n=20,R=R,K=K,B=B,alpha=.05,method=method,se=se,sample_sha256=digest)
                row.update({key:value[bi,mi].item() for key,value in metrics.items()});rows.append(row)
        progress["observed_original_count"]=i+1
    check();data["reason_labels"]=np.array(labels,dtype="U96")
    return data,rows


def index_originals(rows):
    index={}
    for r in rows:
        group=index.setdefault((r["distribution"],r["master_seed"],r["B"],r["method"]),{})
        if r["outer_id"] in group or r["outer_id"] not in range(200):raise ValueError("Duplicate/out-of-range original IDs")
        group[r["outer_id"]]=r
    return index


def expected_values(index,dist,seed,B,method,metric):
    group=index.get((dist,seed,B,method),{})
    return np.array([group[i][metric] if i in group and group[i]["complete"] else np.nan for i in range(200)])


def _values(index,dist,seeds,B,method,metric):
    return np.concatenate([expected_values(index,dist,s,B,method,metric) for s in seeds])


def metric_statistics(values):
    values=np.asarray(values,dtype=float);stats=mean_t_interval(values,confidence=.99)
    quantiles=np.quantile(values,QUANTILES,method="linear") if len(values) and np.isfinite(values).all() else np.full(5,np.nan)
    return dict(mean=stats["mean"],sd=stats["sd"],mcse=stats["mcse"],decision_low=stats["low"],decision_high=stats["high"],
                **{f"p{int(q*100):02d}":float(value) for q,value in zip(QUANTILES,quantiles)})


def summarize(rows,pooled=False):
    index=index_originals(rows);output=[]
    for dist in DISTRIBUTIONS:
        for seed in (["pooled"] if pooled else (0,1,2)):
            seeds=(0,1,2) if pooled else (seed,);planned=600 if pooled else 200
            for B in (999,4999):
                for method in METHODS:
                    group=[r for s in seeds for r in index.get((dist,s,B,method),{}).values()];complete=[r for r in group if r["complete"]]
                    row=dict(distribution=dist,master_seed=seed,n=20,R=planned,K=30,B=B,alpha=.05,method=method,original_total=planned,observed_original_count=len(group),missing_original_count=planned-len(group),
                             complete_original_count=len(complete),incomplete_original_count=planned-len(complete),interval_count=sum(r["total_inner"] for r in group),valid_count=sum(r["valid_inner"] for r in group),invalid_count=sum(r["invalid_inner"] for r in group),conditional_original_count=len(complete))
                    for metric in METRICS:
                        values=_values(index,dist,seeds,B,method,metric)
                        for prefix,array in (("",values),("conditional_",values[np.isfinite(values)])):
                            for suffix,value in metric_statistics(array).items():row[f"{prefix}{metric}"+("" if suffix=="mean" else "_"+suffix)]=value
                    output.append(row)
    return output


def population_contrasts(rows):
    index=index_originals(rows);output=[]
    for seed in (0,1,2,"pooled"):
        seeds=(0,1,2) if seed=="pooled" else (seed,)
        for B in (999,4999):
            for method in METHODS:
                row=dict(master_seed=seed,B=B,method=method,n=20,K=30,alpha=.05,group1="lognormal",group0="gamma",independent_groups=True)
                for metric in METRICS:
                    one,zero=(_values(index,d,seeds,B,method,metric) for d in ("lognormal","gamma"))
                    for prefix,a,b in (("",one,zero),("conditional_",one[np.isfinite(one)],zero[np.isfinite(zero)])):
                        for key,value in welch_interval(a,b,confidence=.99).items():row[f"{prefix}{metric}_{key}"]=value
                output.append(row)
    return output


def paired_contrasts(rows,kind):
    index=index_originals(rows);output=[]
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2,"pooled"):
            seeds=(0,1,2) if seed=="pooled" else (seed,)
            pairs=[(method,method,4999,999) for method in METHODS] if kind=="B" else [(m,r,B,B) for B in (999,4999) for m,r in METHOD_PAIRS]
            for method,reference,B,reference_B in pairs:
                row=dict(distribution=dist,master_seed=seed,method=method,reference=reference,B=B,reference_B=reference_B,n=20,K=30,alpha=.05,original_total=600 if seed=="pooled" else 200)
                for metric in METRICS:
                    values=_values(index,dist,seeds,B,method,metric)-_values(index,dist,seeds,reference_B,reference,metric)
                    for prefix,array in (("",values),("conditional_",values[np.isfinite(values)])):
                        for key,value in mean_t_interval(array,confidence=.99).items():row[f"{prefix}{metric}_{key}"]=value
                if kind=="B":
                    low=_values(index,dist,seeds,999,method,"Ase");high=_values(index,dist,seeds,4999,method,"Ase")
                    mean_low=float(np.mean(low));mean_high=float(np.mean(high))
                    row.update(mean_Ase_999=mean_low,mean_Ase_4999=mean_high,ratio_mean_Ase_4999_over_999=mean_high/mean_low if np.isfinite([mean_low,mean_high]).all() and mean_low>0 else np.nan)
                output.append(row)
    return output


def _reduction_quantity(index,dist,seeds):
    low=_values(index,dist,seeds,999,"bca","Ase");high=_values(index,dist,seeds,4999,"bca","Ase")
    D=high-.75*low;stats=mean_t_interval(D,confidence=.99)
    return {**stats,**reduction_status(stats["low"],stats["high"],complete=bool(stats["valid"])),
            "conditional_interval":mean_t_interval(D[np.isfinite(D)],confidence=.99),"mean_Ase_999":float(np.mean(low)),"mean_Ase_4999":float(np.mean(high)),
            "ratio_mean_Ase_4999_over_999":float(np.mean(high)/np.mean(low)) if np.isfinite(low).all() and np.isfinite(high).all() and np.mean(low)>0 else np.nan}


def assemble_hypotheses(rows):
    index=index_originals(rows);one,zero=(_values(index,d,(0,1,2),4999,"bca","Ase") for d in ("lognormal","gamma"))
    h1=welch_interval(one,zero,confidence=.99)
    h1.update(practical_difference_status(h1["low"],h1["high"],margin=.01,complete=bool(h1["valid"])))
    h1.update(metric="Ase",margin=.01,units="sample-SE-normalized endpoint variability, not percentage points",group1="lognormal",group0="gamma",independent_groups=True,
              conditional_welch=welch_interval(one[np.isfinite(one)],zero[np.isfinite(zero)],confidence=.99))
    reductions={dist:_reduction_quantity(index,dist,(0,1,2)) for dist in DISTRIBUTIONS}
    seed_details=[]
    for seed in (0,1,2):
        seed_details.append(dict(master_seed=seed,H1=welch_interval(expected_values(index,"lognormal",seed,4999,"bca","Ase"),expected_values(index,"gamma",seed,4999,"bca","Ase"),confidence=.99),
                                 **{f"H2_{dist}":_reduction_quantity(index,dist,(seed,)) for dist in DISTRIBUTIONS}))
    return dict(question_id="RQ-M06",confirmatory_intervals=3,decision_confidence=.99,independent_originals_per_population=600,target_method="bca",seed_direction_gate=False,
                quantities=dict(H1=h1,**{f"H2_{dist}":q for dist,q in reductions.items()}),seed_details=seed_details,
                hypotheses=dict(H1={key:h1[key] for key in ("status","direction","equivalence_supported","reason")},
                                H2=dict(status=combine_reduction_status([q["status"] for q in reductions.values()]),**{dist:q["status"] for dist,q in reductions.items()})))


def save_panel(path,panel,dist,seed,K,Bs,planned=200):
    with path.open("xb") as handle:
        np.savez_compressed(handle,**panel,methods=np.array(METHODS),B_values=np.array(Bs),inner_id=np.arange(K,dtype=np.uint8),distribution=dist,master_seed=seed,n=20,R=planned,observed_original_count=len(panel["outer_id"]),K=K,alpha=.05,namespace=MAIN_NAMESPACE,input_accepted_head=ACCEPTED_HEAD)


def write_figure(pooled,hypotheses,output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    output.mkdir(exist_ok=False);fig=make_subplots(rows=1,cols=2,subplot_titles=["Mean BCa Ase: endpoint variability / sample SE","H1 independent mean Ase difference L-G; +/-0.01 band"])
    for dist in DISTRIBUTIONS:
        values=[next(r for r in pooled if r["distribution"]==dist and r["method"]=="bca" and r["B"]==B) for B in (999,4999)]
        fig.add_trace(go.Scatter(x=["999","4999"],y=[r["Ase"] for r in values],mode="lines+markers",name=dist,error_y=dict(type="data",symmetric=False,array=[r["Ase_decision_high"]-r["Ase"] for r in values],arrayminus=[r["Ase"]-r["Ase_decision_low"] for r in values])),row=1,col=1)
    q=hypotheses["quantities"]["H1"]
    fig.add_trace(go.Scatter(x=[q["difference"]],y=["BCa B4999"],mode="markers",name="L-G",error_x=dict(type="data",symmetric=False,array=[q["high"]-q["difference"]],arrayminus=[q["difference"]-q["low"]])),row=1,col=2)
    fig.add_vrect(x0=-.01,x1=.01,fillcolor="gray",opacity=.15,line_width=0,row=1,col=2);fig.add_vline(x=0,line_dash="dot",row=1,col=2)
    fig.update_layout(template="plotly_white",width=1250,height=600,title="RQ-M06: reused saved originals, n20/K30; 99% model-based original-level intervals, N600/population")
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
    attempt=dict(question_id="RQ-M06",status="started",started_utc=datetime.now(timezone.utc).isoformat(),source_git_sha=source_commit,manifest_sha256=sha256(manifest_path),namespace=MAIN_NAMESPACE,single_main_attempt=True)
    _write_json(output/"attempt.json",attempt);(output/"panels").mkdir();(output/"streams").mkdir()
    rows,completed=[],[];resources={**attempt,"environment":runtime_environment(),"R":200,"K":30,"B_values":[999,4999],"n":20,"total_cells":6,
                                 "mother_samples_generated":0,"planned_audit_streams":18,"all_other_bootstrap_arrays_saved":False,"max_resample_elements":1000000,
                                 "wall_scope":"module start before heavy imports through outputs, excluding interpreter startup/final resource write; guard/time separately measure full CLI"}
    def check():enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output))
    def audit_sink(payload):
        path=output/"streams"/f"{payload['distribution']}_seed{payload['master_seed']}_outer{payload['outer_id']}_inner{payload['inner_id']}.npz"
        with path.open("xb") as handle:np.savez_compressed(handle,**payload)
    try:
        for dist,seed in fixed_cells():
            resources["active_cell"]=[dist,seed];progress={}
            try:panel,cell_rows=run_cell(dist,seed,inputs[(dist,seed)],check=check,audit_sink=audit_sink,progress=progress)
            except Exception:
                observed=progress.get("observed_original_count",0);resources["partial_original_count"]=observed
                if observed:
                    partial={key:array[:observed] for key,array in progress["panel"].items()};partial["reason_labels"]=np.array(progress["reason_labels"],dtype="U96")
                    save_panel(output/"panels"/f"{dist}_seed{seed}_partial.npz",partial,dist,seed,30,(999,4999));rows.extend(progress["rows"])
                raise
            save_panel(output/"panels"/f"{dist}_seed{seed}.npz",panel,dist,seed,30,(999,4999));rows.extend(cell_rows);completed.append([dist,seed]);check()
        write_csv(output/"original_metrics.csv",rows);summary,pooled=summarize(rows),summarize(rows,pooled=True)
        write_csv(output/"summary.csv",summary);write_csv(output/"pooled_summary.csv",pooled);write_csv(output/"population_contrasts.csv",population_contrasts(rows))
        write_csv(output/"paired_B_summary.csv",paired_contrasts(rows,"B"));write_csv(output/"paired_method_summary.csv",paired_contrasts(rows,"method"))
        hypotheses=assemble_hypotheses(rows);_write_json(output/"hypotheses.json",hypotheses);write_figure(pooled,hypotheses,output/"figures")
        check();resources["status"]="completed"
    except Exception as exc:
        resources.update(status="stopped",stop_reason=type(exc).__name__,stop_detail=str(exc) if isinstance(exc,BudgetExceeded) else "See preserved diagnostics; no automatic retry")
        if rows and not (output/"original_metrics.csv").exists():write_csv(output/"original_metrics.csv",rows)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGTERM,signal.SIG_IGN)
        resources.update(completed_cells=len(completed),completed_cell_keys=completed,incomplete_cells=[list(c) for c in fixed_cells() if list(c) not in completed],saved_audit_streams=len(list((output/"streams").glob("*.npz"))),finished_utc=datetime.now(timezone.utc).isoformat());_final_resources(output,resources)
    return resources


def run_smoke(output):
    inputs_provenance=make_inputs_manifest(root=ROOT);inputs=load_inputs(inputs_provenance,root=ROOT);output=output.resolve()
    if ROOT not in output.parents:raise ValueError("Smoke output must be inside project")
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError("Existing smoke attempt preserved")
    attempt=dict(question_id="RQ-M06",scope="smoke only; saved originals, no scientific decisions",R=2,K=2,B_values=[99],namespace=SMOKE_NAMESPACE,source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})
    _write_json(output/"attempt.json",attempt);result,completed={**attempt,"environment":runtime_environment(),"status":"started"},[]
    try:
        for dist,seed in fixed_cells():
            panel,rows=run_cell(dist,seed,inputs[(dist,seed)],R=2,K=2,Bs=(99,),namespace=SMOKE_NAMESPACE,check=lambda:enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output)))
            completed.append(dict(distribution=dist,master_seed=seed,valid_entries=int(panel["valid"].sum())))
        result["status"]="completed"
    except Exception as exc:result.update(status="stopped",stop_reason=type(exc).__name__);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGTERM,signal.SIG_IGN)
        result.update(completed_cells=len(completed),cells=completed,mother_samples_generated=0,saved_originals_used=2*len(completed),inner_streams=4*len(completed),bootstrap_samples=396*len(completed),logical_resampling_indices=7920*len(completed));_final_resources(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-manifest",action="store_true");modes.add_argument("--run",action="store_true");modes.add_argument("--smoke",action="store_true")
    parser.add_argument("--manifest",type=Path,default=ROOT/"evidence/rq_m06/manifest.json");parser.add_argument("--output",type=Path);parser.add_argument("--source-commit")
    args=parser.parse_args()
    if args.prepare_manifest:
        path=args.manifest.resolve()
        if ROOT not in path.parents:parser.error("Manifest must be inside project")
        path.parent.mkdir(parents=True,exist_ok=True);_write_json(path,make_manifest());print("Prepared",path.relative_to(ROOT))
    elif args.smoke:
        result=run_smoke(args.output or ROOT/"evidence/rq_m06/smoke");print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))
    else:
        if not args.source_commit:parser.error("--run requires full reviewed --source-commit")
        result=run_benchmark(args.manifest.resolve(),args.output or ROOT/"evidence/rq_m06",args.source_commit);print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))


if __name__=="__main__":main()
