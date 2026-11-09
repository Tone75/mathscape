"""Preregister and run RQ-M08: three allocations share one120-stream budget.

Only bootstrap indices from accepted saved originals can be generated here.
Main and the sole specified smoke are launched by Root, never by tests/UI/audit.
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


class BudgetExceeded(RuntimeError):pass


def _deadline(signum,frame):raise BudgetExceeded("294-second worker soft stop; preserve partial diagnostics")


if __name__=="__main__" and ("--run" in sys.argv or "--smoke" in sys.argv):
    signal.signal(signal.SIGALRM,_deadline);signal.signal(signal.SIGTERM,_deadline)
    signal.setitimer(signal.ITIMER_REAL,max(.001,294-(perf_counter()-_PROCESS_START)))

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from mathscape.allocation import (
    make_inputs_manifest,validate_inputs_manifest,load_inputs,generate_bootstrap_means,
    prepare_sample,infer_single,original_metrics,mean_t_interval,band_status,variance_band_status,joint_status,
    ALLOCATION_METRICS,GROUP_FIELDS,DECOMPOSITION_FIELDS,NUMERIC_RTOL,NUMERIC_ATOL,IDENTITY_RTOL,IDENTITY_ATOL,
)
from scripts.run_bootstrap_benchmark import environment,sha256,peak_rss_bytes,directory_bytes,_write_json,write_csv
from scripts.run_stability_benchmark import sample_hash

DISTRIBUTIONS=("gamma","lognormal")
METHODS=("t","percentile","bca")
ALLOCATIONS=("P30","P60","P120")
MAIN_NAMESPACE,SMOKE_NAMESPACE=0x4D3038,0x534D3038
ACCEPTED_HEAD="39bcdc367320b6da513df72d2fa43bfdb7f060f4"
DESIGN=dict(distributions=list(DISTRIBUTIONS),n=20,master_seeds=[0,1,2],originals_per_cell=10,R=20,
            streams_per_budget=120,allocations=[dict(allocation=p,J=j,K=k) for p,j,k in zip(ALLOCATIONS,(4,2,1),(30,60,120))],
            B=4999,alpha=.05,outer_ids=list(range(10)),methods=list(METHODS),input_accepted_head=ACCEPTED_HEAD,source_question="RQ-M07")
POLICY=dict(wall_seconds=300,internal_stop_seconds=294,peak_rss_bytes=2*1024**3,output_bytes=50*1024**2,max_resample_elements=1000000,
            public_overhead_reserve_bytes=4*1024**2,cell_cleanup_reserve_bytes=8*1024**2,decision_confidence=.995,confirmatory_intervals=6,
            mean_movement_margin=.0025,variance_ratio_low=.8,variance_ratio_high=1.25)
SOURCES=("mathscape/allocation.py","mathscape/finite_k.py","mathscape/matched_stability.py","mathscape/matched_moments.py","mathscape/stability.py","mathscape/bootstrap.py","mathscape/inference.py",
         "scripts/run_allocation_benchmark.py","scripts/run_allocation_guarded.py","scripts/audit_allocation.py","scripts/run_bootstrap_benchmark.py","scripts/run_stability_benchmark.py",
         "tests/test_allocation.py","tests/test_allocation_runner.py","tests/test_allocation_guard.py","tests/test_allocation_audit.py","requirements.txt","evidence/rq_m08/SPEC.md")
DIAGNOSTICS=("lower","upper","z0","a","q_low","q_high","tau_low","tau_high")
QUANTILES=(.05,.25,.50,.75,.95)
SUMMARY_METRICS=("M","Q","Q_over_R","T","Vbar","lower_sd","upper_sd","S","Ase","Apop","Vse","C","Mbelow","Mabove","miss_T","mean_width","warning_rate")
GAPS=("Jgap","Pgap","delta")


def runtime_environment():return {**environment(),"streamlit":version("streamlit"),"pytest":version("pytest")}


def fixed_cells():return [(d,s) for d in DISTRIBUTIONS for s in (0,1,2)]


def make_manifest():
    return dict(schema_version=1,question_id="RQ-M08",created_utc=datetime.now(timezone.utc).isoformat(),design=DESIGN,inputs=make_inputs_manifest(root=ROOT),
        totals=dict(original_samples=60,originals_per_population=30,input_cells=6,budget_repeats=1200,inner_streams=144000,bootstrap_samples=719856000,
                    logical_resampling_indices=14397120000,logical_intervals=432000,group_rows=25200,budget_rows=10800,original_allocation_rows=540,
                    original_comparison_rows=180,seed_summary_rows=54,pooled_summary_rows=18),
        input_contract="accepted M07 G/L seed0,1,2 outer0..9 exact rawbytes/hash/IDs, M06/M05 lineage verified; no selection by outcomes, new originals, standardization or old endpoints",
        disclosure="M07 aggregate/decomposition known at design; reused60 originals and new namespace streams are not an independent-original replication; no update/retest of accepted earlier RQs",
        randomness=dict(generator="Generator(PCG64(SeedSequence(...)))",entropy="saved master_seed",main_namespace=MAIN_NAMESPACE,smoke_namespace=SMOKE_NAMESPACE,
            distribution_ids=dict(gamma=1,lognormal=2),spawn_key=["namespace","distribution_id","n","outer_id","purpose_id","budget_repeat_id","stream_id"],purpose_ids=dict(bootstrap=1),
            budget_repeat_ids=list(range(20)),stream_ids=list(range(120)),outer_ids=list(range(10)),native_indices_dtype="int64",stored_indices_dtype="uint8",
            excluded_from_key=["B","J","K","allocation","method","execution_order","batch_size"],
            sharing="one new120-stream set per original/budget repeat; contiguous nonoverlapping groups4x30/2x60/1x120 share all120 streams; same means for percentile/BCa; t/a/se once original"),
        inference="unchanged accepted ddof1,linear quantile,half-tie z0,jackknife a and q/tau validity; no truth/population input, clipping,fallback,adaptiveB or warning exclusions",
        metrics=dict(V="(sample_var(lower)+sample_var(upper))/(2*sample_SE^2),ddof1",A="sqrt(V)",T="mean_j(sqrt(V_j)), never sqrt(mean_j(V_j))",M="mean_r(T)",Q="sample_var_r(T),ddof1",
            allocation_aliases="Vse=Vbar=mean_j(Vgroup);Ase=T=mean_j(sqrt(Vgroup));Vse generally differs from T^2 forJ>1;lower_sd/upper_sd/S average groupSD/S,not pooledendpointSD",
            d="mean_r(T_P120-T_P30)",L="Q_P120-.8*Q_P30",U="Q_P120-1.25*Q_P30",Q_over_R="descriptive conditional variance estimate for meanM only, not primaryQ",
            decomposition="T_P120-T_P30=Jgap+Pgap;Jgap=sqrt(mean4V30)-mean4sqrtV30;Pgap=sqrtV120-sqrt(mean4V30);no M07 first-group Rgap",
            ss_identity="119*var120=29*sum4(var30)+30*sum4((mean30-mean120)^2), and119*var120=59*sum2(var60)+60*sum2((mean60-mean120)^2),both endpoints",
            interpretation="equal resampling work not equal wall; finiteK120 not truth; no absolute bias/RMSE/calibration/optimalallocation claim; Jgap nonnegative is algebra not new hypothesis",
            t_control="computed once and broadcast; A/T/M/Q/d/L/U zero only if all required logical slots valid; count one independent interval per original, not R*streams/allocations; ratios/comparisonSE0 NA; oldK1NA unchanged",
            summaries="N10/seed,N30/pop original-level mean/SD/MCSE/t99.5 intervals and linear p05/25/50/75/95; R/J/K never increaseN; conditional_ references explicit",
            ratios="mean_individual_Q120_Q30 versus ratio_mean_Q120_over_mean_Q30 distinct descriptive references; zero denominatorNA includingt; H2 uses meanL/meanU and positive finite meanQ30"),
        confirmation=dict(H1="BCa P120-P30 mean d per population t99.5 df29; strict inside(-.0025,.0025) support,lo>.0025 orhi<-.0025 not_supported,otherwisehold",
            H2="positive finite meanQ30 required; meanL99.5low>0 AND meanU99.5high<0 support; meanLhigh<=0 OR meanUlow>=0 not_supported; elsehold; bothcomparisonsdefinedrequired",
            joint="bothpopulation support=>support;anynot_supported=>not_supported;elsehold; H1&H2 both support needed for simultaneous mean/variance-equivalence wording",
            intervals=6,confidence=.995,seed_gate=False,P60="descriptive only",units="H1 Ase units not percentagepoints; H2 multiplicative variance ratio(.8,1.25),not +/-25% orSD ratio",
            completeness="allR20*120 required peroriginal/method; anymissing/invalid/nonfinite/nonpositiveSE => primaryNA keepN30; primarySE0=>hold; unrelatedmethod invalid doesnotpoison BCa; conditional never decides",
            approximation="original-level approximate t on IID originals selectedbyID; reusedsamples/finiteR,K/non-normalN30 limitations; not conditional inner-randomness-onlyCI; sixnominal99.5 intervals imply approximate union noncoverage<=3%,no exactguarantee"),
        numeric_tolerances=dict(numeric_rtol=NUMERIC_RTOL,numeric_atol=NUMERIC_ATOL,identity_rtol=IDENTITY_RTOL,identity_atol=IDENTITY_ATOL,
            identity_rule="abs(residual)<=atol+rtol*scale;SSscale=max(abs(lhs),abs(rhs));normalized=residual/max(1,scale)",provenance="inheritedM07 sameidentity comparisons,deterministicfixtureslockedbeforemain;no outcome-based relaxation"),
        storage=dict(panel_axes=["original","budget_repeat","method","stream"],original_count_symbol="O; R means20 budget repeats, J onlyallocation4/2/1",
            diagnostics="float64lowerupperz0aqtau,boolvalid/tailwarning/observed_streams(O,R,stream);uint8reason/U96dictionary;samplefloat64/hashS64/outeruint16/budget_repeat&streamuint8",
            panel_metadata=["O","R","streams_per_budget","B","n","alpha","namespace","input_accepted_head","methods","budget_repeat_id","stream_id","observed_original_count","completed_original_count","observed_stream_count"],
            tables={"group_metrics.csv.gz":25200,"budget_metrics.csv.gz":10800,"decomposition_metrics.csv.gz":3600,"original_allocation_metrics.csv":540,"original_comparison_metrics.csv":180,"summary.csv":54,"pooled_summary.csv":18,"decomposition_summary.csv":24},
            table_contract=dict(group_fields=list(GROUP_FIELDS),allocation_fields=list(ALLOCATION_METRICS),decomposition_fields=list(DECOMPOSITION_FIELDS),summary_fields=list(SUMMARY_METRICS),
                group_order="P30groups0..3:0..29/30..59/60..89/90..119;P60groups0..1:0..59/60..119;P120group0:0..119",
                counts="corestrict total_streams/valid_streams/invalid_streams;all120requiredpermethod;effective t1peroriginal and1perbudget;allocationintervalcounts reference same120sharedstreamset,never add3allocationsasgeneration",
                coverage_name="miss_T=Mbelow-Mabove, distinct from budget estimatorT",original_comparison_fields="d,L,U,Q30,Q120,M30,M120,ratio_Q120_Q30,meanJ/P/delta,maxabsresiduals,complete/observedcounts;noRgap"),
            selected_streams=[dict(distribution=d,master_seed=s,outer_id=0,budget_repeat_id=0,stream_id=k) for d,s in fixed_cells() for k in (0,29,30,59,60,119)],
            stream_schema="exact36indicesuint8(4999,20) reversiblystoredfromnativeint64+meansfloat64(4999),IDs/hash/metadata; other143964 sequences not reconstructed",
            estimates=dict(raw_endpoint_numeric_bytes=27648000,raw_mask_reason_bytes=1296000,raw_original_bytes=9600,raw_selected_stream_bytes=5038992,
                panel_reserve_bytes=29*1024**2,stream_reserve_bytes=5*1024**2,csv_reserve_bytes=5*1024**2,single_html_reserve_bytes=5*1024**2,other_public_reserve_bytes=4*1024**2,total_public_reserve_bytes=48*1024**2),
            estimate_caveat="conservativeplannedupperallocation; prewrite guards assume rawarray sizes+headers,no compression; actualresourceguard stopswithoutscientificresizing"),
        policy=POLICY,execution=dict(main="once rootguardedCLIonly;tests/UI/auditneverlaunch",smoke="sole6savedouter0/R2/common12/B99 groups4x3/2x6/1x12,144streams14256resamples285120indices separatenamespace,not science",
            stop="worker294,parentTERM295/KILL298/finish299 externalcap300; preserveactiveoriginal partial+observedmask and priorresults,noauto retry/shrink",correction="onlyactualcodebug,maxone separatelyauthorized attempt retainingoldsource/results/reason,cumulative mainwall<=600"),
        timing="sole combinedrun instrumentation:cachedcontext+streamgeneration/inference/endpointstorage wall;allocationaggregation/flatten wall peroriginal;remaining setup/guards/summary/output wall; saved_stream_output_seconds is a SUBSET of remaining_setup_guard_summary_output_seconds,never a fourthdisjointphase;not independentlymeasured perallocation timings; fullCLI externaltime includesstartup/exit",
        environment=runtime_environment(),source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})


def validate_manifest(m):
    if m.get("schema_version")!=1 or m.get("question_id")!="RQ-M08" or m.get("design")!=DESIGN or m.get("policy")!=POLICY:raise ValueError("Manifest differs from fixed RQ-M08")


def verify_committed_inputs(path,m,commit):
    if len(commit)!=40 or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=commit:raise ValueError("--source-commit must be full current reviewed HEAD")
    for name in [*SOURCES,str(path.resolve().relative_to(ROOT))]:
        if subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=ROOT)!=(ROOT/name).read_bytes():raise ValueError(f"Committed source differs: {name}")
    for name in SOURCES:
        if sha256(ROOT/name)!=m["source_files_sha256"].get(name):raise ValueError(f"Preregistered source differs: {name}")
    if runtime_environment()!=m["environment"]:raise ValueError("Runtime environment differs")
    validate_inputs_manifest(m["inputs"],root=ROOT)


def enforce_budget(wall,rss,output):
    if wall>=294 or rss>=POLICY["peak_rss_bytes"] or output+POLICY["public_overhead_reserve_bytes"]+65536>=POLICY["output_bytes"]:raise BudgetExceeded("wall/RSS/output budget reached includingpublicationreserve")


def _begin_cleanup():
    signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGTERM,signal.SIG_IGN)


def require_output_capacity(output,pending_bytes):
    if directory_bytes(output)+pending_bytes+POLICY["public_overhead_reserve_bytes"]+65536>=POLICY["output_bytes"]:raise BudgetExceeded("Output capacity reserved before write")


def npz_size_bound(payload):return int(sum(np.asarray(v).nbytes+1024 for v in payload.values())*1.02)+65536


def _allocate(saved,O,R,streams):
    shape=(O,R,3,streams)
    return {**{key:np.full(shape,np.nan) for key in DIAGNOSTICS},"valid":np.zeros(shape,bool),"tail_warning":np.zeros(shape,bool),"reason_codes":np.zeros(shape,np.uint8),
        "observed_streams":np.zeros((O,R,streams),bool),"samples":np.array(saved["samples"][:O],dtype=float,copy=True),"sample_sha256":np.array(saved["sample_sha256"][:O],dtype="S64"),"outer_id":np.array(saved["outer_id"][:O],dtype=np.uint16)}


def derived_tables(result,raw,se,dist,seed,outer,digest,O,R,streams,B=4999):
    """Losslessly flatten core arrays; counts come from the strict core masks."""
    meta=dict(distribution=dist,master_seed=seed,outer_id=outer,n=20,originals_per_cell=O,R=R,streams_per_budget=streams,B=B,alpha=.05,se=se,sample_sha256=digest)
    output={key:[] for key in ("original_allocations","original_comparisons","budgets","groups","decompositions")}
    budget=result["budget"]
    for pi,p in enumerate(ALLOCATIONS):
        allocation_meta=dict(allocation=p,J=(4,2,1)[pi],K=(streams//4,streams//2,streams)[pi])
        for mi,method in enumerate(METHODS):
            row={**meta,**allocation_meta,"method":method}
            row.update({key:np.asarray(value)[pi,mi].item() for key,value in result["original_allocations"].items()});output["original_allocations"].append(row)
    for mi,method in enumerate(METHODS):
        row={**meta,"method":method};row.update({key:np.asarray(value)[mi].item() for key,value in result["original_comparisons"].items()});output["original_comparisons"].append(row)
    for repeat in range(R):
        for pi,p in enumerate(ALLOCATIONS):
            for mi,method in enumerate(METHODS):
                row={**meta,"method":method,"budget_repeat_id":repeat,"allocation":p,"J":(4,2,1)[pi],"K":(streams//4,streams//2,streams)[pi]}
                row.update({key:np.asarray(value)[repeat,pi,mi].item() for key,value in budget["allocations"].items()});output["budgets"].append(row)
        for gi,pi in enumerate((0,0,0,0,1,1,2)):
            local_group=gi if pi==0 else gi-4 if pi==1 else 0
            size=(streams//4,streams//2,streams)[pi]
            for mi,method in enumerate(METHODS):
                row={**meta,"method":method,"budget_repeat_id":repeat,"allocation":ALLOCATIONS[pi],"group":local_group,"group_index":gi,"J":(4,2,1)[pi],"K":size,"group_start":local_group*size,"group_stop":(local_group+1)*size}
                row.update({key:np.asarray(value)[repeat,mi,gi].item() for key,value in budget["groups"].items()});output["groups"].append(row)
        for mi,method in enumerate(METHODS):
            row={**meta,"method":method,"budget_repeat_id":repeat}
            row.update({key:np.asarray(value)[repeat,mi].item() for key,value in budget["decomposition"].items()})
            for gap in GAPS:row[gap+"_sign"]=int(np.sign(row[gap])) if np.isfinite(row[gap]) else "NA"
            output["decompositions"].append(row)
    return output


def run_cell(dist,seed,saved,O=10,R=20,streams=120,B=4999,namespace=MAIN_NAMESPACE,check=lambda:None,bootstrap_fn=None,audit_sink=None,progress=None):
    for value,low,high in ((O,1,10),(R,2,20),(streams,12,120),(B,1,4999)):
        if not isinstance(value,int) or isinstance(value,bool) or not low<=value<=high:raise ValueError("Invalid fixed-panel fixture dimensions")
    if streams%4 or np.asarray(saved["samples"]).shape[1:]!=(20,) or len(saved["samples"])<O or np.asarray(saved["outer_id"][:O]).tolist()!=list(range(O)):raise ValueError("Saved input shape/IDs or stream grouping differ")
    data=_allocate(saved,O,R,streams);labels=[""];tables={key:[] for key in ("original_allocations","original_comparisons","budgets","groups","decompositions")};bootstrap_fn=bootstrap_fn or generate_bootstrap_means
    progress=progress if progress is not None else {};progress.update(panel=data,reason_labels=labels,completed_original_count=0,retained_original_count=0,observed_stream_count=0,tables=tables,
        endpoint_generation_inference_seconds=0.,saved_stream_output_seconds=0.,allocation_aggregation_flatten_seconds=0.)
    for oi,outer in enumerate(data["outer_id"]):
        progress.update(active_original_id=int(outer),retained_original_count=oi+1)
        check();sample=data["samples"][oi];digest=sample_hash(sample)
        if digest!=data["sample_sha256"][oi].decode("ascii"):raise ValueError("Saved original hash mismatch")
        context_start=perf_counter();context=prepare_sample(sample,alpha=.05);se=float(context["se"])
        progress["endpoint_generation_inference_seconds"]+=perf_counter()-context_start
        for repeat in range(R):
            for stream in range(streams):
                check();stream_start=perf_counter();audit_seconds=0.;failure="";audit=namespace==MAIN_NAMESPACE and int(outer)==0 and repeat==0 and stream in (0,29,30,59,60,119) and audit_sink is not None
                if not context["valid"]:means=np.full(B,np.nan)
                else:
                    try:
                        generated=bootstrap_fn(sample,dist,n=20,master_seed=seed,outer_id=int(outer),budget_repeat_id=repeat,stream_id=stream,Bmax=B,namespace=namespace,max_elements=1000000,return_indices=audit)
                        if audit:
                            means,indices=generated;audit_start=perf_counter();audit_sink(dict(indices=indices,means=means,distribution=dist,master_seed=seed,outer_id=int(outer),budget_repeat_id=repeat,stream_id=stream,n=20,B=B,R=R,streams_per_budget=streams,alpha=.05,namespace=namespace,native_indices_dtype="int64",stored_indices_dtype="uint8",sample_sha256=digest.encode("ascii")));audit_seconds=perf_counter()-audit_start
                        else:means=generated
                    except ValueError as exc:
                        if str(exc)!="bootstrap_mean_nonfinite":raise
                        means=np.full(B,np.nan);failure=str(exc)
                records=infer_single(context,means)
                for mi,method in enumerate(METHODS):
                    record=records[method]
                    for field in DIAGNOSTICS:data[field][oi,repeat,mi,stream]=record[field]
                    data["valid"][oi,repeat,mi,stream]=record["valid"];data["tail_warning"][oi,repeat,mi,stream]=record["tail_warning"]
                    reason=failure if failure and method!="t" else record["reason"]
                    if reason not in labels:
                        if len(labels)>=256 or len(reason)>96:raise ValueError("Reason dictionary capacity exceeded")
                        labels.append(reason)
                    data["reason_codes"][oi,repeat,mi,stream]=labels.index(reason)
                data["observed_streams"][oi,repeat,stream]=True;progress["observed_stream_count"]+=1
                progress["endpoint_generation_inference_seconds"]+=perf_counter()-stream_start-audit_seconds;progress["saved_stream_output_seconds"]+=audit_seconds
        raw={key:value[oi] for key,value in data.items() if key in (*DIAGNOSTICS,"valid","tail_warning")}
        aggregation_start=perf_counter()
        result=original_metrics(raw,se,group_size=streams//4,expected_repeats=R)
        local=derived_tables(result,raw,se,dist,seed,int(outer),digest,O,R,streams,B=B)
        for key in tables:tables[key].extend(local[key])
        progress["allocation_aggregation_flatten_seconds"]+=perf_counter()-aggregation_start
        progress["completed_original_count"]=oi+1
    check();data["reason_labels"]=np.array(labels,dtype="U96")
    return data,tables


def _index(rows,allocation=False):
    result={}
    for row in rows:
        key=(row["distribution"],row["master_seed"],row["method"])+((row["allocation"],) if allocation else ())
        group=result.setdefault(key,{})
        if row["outer_id"] in group or row["outer_id"] not in range(10):raise ValueError("Duplicate/out-of-range original ID")
        group[row["outer_id"]]=row
    return result


def _expected(index,dist,seed,method,field,allocation=None):
    group=index.get((dist,seed,method)+((allocation,) if allocation else ()),{})
    return np.array([group[i][field] if i in group and group[i]["complete"] else np.nan for i in range(10)],dtype=float)


def _values(index,dist,seeds,method,field,allocation=None):return np.concatenate([_expected(index,dist,s,method,field,allocation) for s in seeds])


def metric_statistics(values):
    values=np.asarray(values,dtype=float);stats=mean_t_interval(values,confidence=.995)
    qs=np.quantile(values,QUANTILES,method="linear") if len(values) and np.isfinite(values).all() else np.full(5,np.nan)
    return dict(mean=stats["mean"],sd=stats["sd"],mcse=stats["mcse"],decision_low=stats["low"],decision_high=stats["high"],**{f"p{int(q*100):02d}":float(v) for q,v in zip(QUANTILES,qs)})


def summarize(original_allocations,pooled=False):
    index=_index(original_allocations,allocation=True);output=[]
    for dist in DISTRIBUTIONS:
        for seed in (["pooled"] if pooled else (0,1,2)):
            seeds=(0,1,2) if pooled else (seed,);planned=30 if pooled else 10
            for method in METHODS:
                for pi,p in enumerate(ALLOCATIONS):
                    group=[r for s in seeds for r in index.get((dist,s,method,p),{}).values()];complete=[r for r in group if r["complete"]]
                    row=dict(distribution=dist,master_seed=seed,n=20,R=20,allocation=p,J=(4,2,1)[pi],K=(30,60,120)[pi],streams_per_budget=120,B=4999,alpha=.05,method=method,
                        original_total=planned,observed_original_count=len(group),missing_original_count=planned-len(group),complete_original_count=len(complete),incomplete_original_count=planned-len(complete),
                        interval_count=sum(r["total_streams"] for r in group),valid_count=sum(r["valid_streams"] for r in group),invalid_count=sum(r["invalid_streams"] for r in group),conditional_original_count=len(complete))
                    for field in SUMMARY_METRICS:
                        values=_values(index,dist,seeds,method,field,p)
                        for prefix,array in (("",values),("conditional_",values[np.isfinite(values)])):
                            for suffix,value in metric_statistics(array).items():row[f"{prefix}{field}"+("" if suffix=="mean" else "_"+suffix)]=value
                    output.append(row)
    return output


class TableWriter:
    def __init__(self,handle):self.handle=handle;self.writer=None
    def write_rows(self,rows):
        for row in rows:
            if self.writer is None:self.writer=csv.DictWriter(self.handle,fieldnames=list(row));self.writer.writeheader()
            self.writer.writerow(row)


@contextmanager
def gzip_table(path):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="",mode="wb",compresslevel=6,mtime=0,fileobj=raw) as zipped:
            with io.TextIOWrapper(zipped,encoding="utf-8",newline="") as text:yield TableWriter(text)


def save_panel(path,panel,dist,seed,R,streams,B,planned=10,capacity_output=None):
    payload=dict(**panel,methods=np.array(METHODS),budget_repeat_id=np.arange(R,dtype=np.uint8),stream_id=np.arange(streams,dtype=np.uint8),distribution=dist,master_seed=seed,n=20,
        O=planned,observed_original_count=len(panel["outer_id"]),completed_original_count=int(np.all(panel["observed_streams"],axis=(1,2)).sum()),observed_stream_count=int(panel["observed_streams"].sum()),
        R=R,streams_per_budget=streams,B=B,alpha=.05,namespace=MAIN_NAMESPACE,input_accepted_head=ACCEPTED_HEAD)
    if capacity_output is not None:require_output_capacity(capacity_output,npz_size_bound(payload))
    with path.open("xb") as handle:np.savez_compressed(handle,**payload)


def write_figure(pooled,output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    output.mkdir(exist_ok=False);fig=make_subplots(rows=1,cols=2,subplot_titles=["BCa mean budget estimate M (original-level99.5% bars)","BCa repeated-budget variance Q"])
    for dist in DISTRIBUTIONS:
        rows=[next(r for r in pooled if r['distribution']==dist and r['method']=='bca' and r['allocation']==p) for p in ALLOCATIONS]
        for col,field in ((1,'M'),(2,'Q')):
            fig.add_trace(go.Scatter(x=list(ALLOCATIONS),y=[r[field] for r in rows],mode='lines+markers',name=dist,legendgroup=dist,showlegend=col==1,
                error_y=dict(type='data',symmetric=False,array=[r[field+'_decision_high']-r[field] for r in rows],arrayminus=[r[field]-r[field+'_decision_low'] for r in rows])),row=1,col=col)
    fig.update_layout(template='plotly_white',width=1250,height=600,title='RQ-M08: BCa6conditions; equal120 streams, B4999,R20,N30/population; P120 finite,not truth')
    fig.write_html(str(output/'overview.html'),include_plotlyjs=True,full_html=True,config={'displaylogo':False})


def _final_resources(output,record):
    record.update(process_wall_seconds=perf_counter()-_PROCESS_START,process_peak_rss_bytes=peak_rss_bytes(),output_bytes_before_resources=directory_bytes(output));record['output_bytes_total']=record['output_bytes_before_resources']
    if 'endpoint_generation_inference_seconds' in record:
        record['remaining_setup_guard_summary_output_seconds']=max(0.,record['process_wall_seconds']-record['endpoint_generation_inference_seconds']-record['allocation_aggregation_flatten_seconds'])
    for _ in range(6):
        size=record['output_bytes_before_resources']+len((json.dumps(record,indent=2,allow_nan=False)+'\n').encode())
        if size==record['output_bytes_total']:break
        record['output_bytes_total']=size
    _write_json(output/'resources.json',record)


def run_benchmark(manifest_path,output,source_commit):
    m=json.loads(manifest_path.read_text());validate_manifest(m);verify_committed_inputs(manifest_path,m,source_commit);inputs=load_inputs(m['inputs'],root=ROOT)
    output=output.resolve()
    if ROOT not in output.parents:raise ValueError('Output must be inside project')
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ('attempt.json','resources.json','panels','streams','original_allocation_metrics.csv','summary.csv','figures')):raise FileExistsError('Existing attempt/results preserved')
    attempt=dict(question_id='RQ-M08',status='started',started_utc=datetime.now(timezone.utc).isoformat(),source_git_sha=source_commit,manifest_sha256=sha256(manifest_path),namespace=MAIN_NAMESPACE,single_main_attempt=True)
    _write_json(output/'attempt.json',attempt);(output/'panels').mkdir();(output/'streams').mkdir()
    allocations,comparisons,decompositions,completed=[],[],[],[];observed_streams,observed_invalid=0,0
    resources={**attempt,'environment':runtime_environment(),'originals_per_cell':10,'R':20,'streams_per_budget':120,'B':4999,'n':20,'allocations':DESIGN['allocations'],'total_cells':6,'mother_samples_generated':0,
        'planned_audit_streams':36,'all_other_bootstrap_arrays_saved':False,'max_resample_elements':1000000,'main_attempts':1,'reruns':0,'planned_originals':60,'planned_inner_streams':144000,'planned_logical_intervals':432000,
        'endpoint_generation_inference_seconds':0.,'allocation_aggregation_flatten_seconds':0.,'saved_stream_output_seconds':0.,
        'timing_scope':'One combined run sharing120 streams across all3allocations. Endpoint generation/inference includes cachedcontext once/original and endpointrecord storage; aggregation/flatten measured once/original; remaining_setup_guard_summary_output_seconds includes imports/preflight/guards/summary/alloutputs. saved_stream_output_seconds is a SUBSET of that remaining phase,not a fourthdisjointphase; not perallocation benchmark.',
        'wall_scope':'module start before heavy imports through outputs excluding interpreter startup/finalresourcewrite; externalguard/time includes fullCLI'}
    size_state=dict(calls=0,size=directory_bytes(output))
    def check(force=False):
        size_state['calls']+=1
        if force or size_state['calls']%128==1:size_state['size']=directory_bytes(output)
        enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),size_state['size'])
    def audit_sink(payload):
        path=output/'streams'/f"{payload['distribution']}_seed{payload['master_seed']}_outer{payload['outer_id']}_repeat{payload['budget_repeat_id']}_stream{payload['stream_id']}.npz"
        require_output_capacity(output,npz_size_bound(payload)+POLICY['cell_cleanup_reserve_bytes'])
        with path.open('xb') as handle:np.savez_compressed(handle,**payload)
        check(force=True)
    def collect_timing(progress):
        for key in ('endpoint_generation_inference_seconds','allocation_aggregation_flatten_seconds','saved_stream_output_seconds'):resources[key]+=progress.get(key,0.)
    try:
        with gzip_table(output/'budget_metrics.csv.gz') as budget_writer,gzip_table(output/'group_metrics.csv.gz') as group_writer,gzip_table(output/'decomposition_metrics.csv.gz') as decomposition_writer:
            for dist,seed in fixed_cells():
                resources['active_cell']=[dist,seed];progress={};require_output_capacity(output,POLICY['cell_cleanup_reserve_bytes'])
                try:panel,tables=run_cell(dist,seed,inputs[(dist,seed)],check=check,audit_sink=audit_sink,progress=progress)
                except Exception:
                    _begin_cleanup();collect_timing(progress)
                    retained=progress.get('retained_original_count',0);observed_streams+=progress.get('observed_stream_count',0)
                    resources.update(partial_completed_original_count=progress.get('completed_original_count',0),partial_retained_original_count=retained,partial_observed_stream_count=progress.get('observed_stream_count',0),partial_active_original_id=progress.get('active_original_id'))
                    if retained:
                        mask=progress['panel']['observed_streams'][:,:,None,:];observed_invalid+=int((~progress['panel']['valid']&mask).sum())
                        partial={key:value[:retained] for key,value in progress['panel'].items()};partial['reason_labels']=np.array(progress['reason_labels'],dtype='U96')
                        save_panel(output/'panels'/f'{dist}_seed{seed}_partial.npz',partial,dist,seed,20,120,4999,capacity_output=output)
                        local=progress['tables'];allocations.extend(local['original_allocations']);comparisons.extend(local['original_comparisons']);decompositions.extend(local['decompositions'])
                        budget_writer.write_rows(local['budgets']);group_writer.write_rows(local['groups']);decomposition_writer.write_rows(local['decompositions'])
                    raise
                collect_timing(progress)
                save_panel(output/'panels'/f'{dist}_seed{seed}.npz',panel,dist,seed,20,120,4999,capacity_output=output)
                observed_streams+=int(panel['observed_streams'].sum());observed_invalid+=int((~panel['valid']).sum())
                allocations.extend(tables['original_allocations']);comparisons.extend(tables['original_comparisons']);decompositions.extend(tables['decompositions'])
                budget_writer.write_rows(tables['budgets']);group_writer.write_rows(tables['groups']);decomposition_writer.write_rows(tables['decompositions']);completed.append([dist,seed]);check(force=True)
        require_output_capacity(output,5*1024**2+5*1024**2)
        write_csv(output/'original_allocation_metrics.csv',allocations);write_csv(output/'original_comparison_metrics.csv',comparisons)
        pooled=summarize(allocations,True);write_csv(output/'summary.csv',summarize(allocations));write_csv(output/'pooled_summary.csv',pooled)
        write_csv(output/'decomposition_summary.csv',decomposition_summary(comparisons,decompositions));_write_json(output/'hypotheses.json',assemble_hypotheses(comparisons));write_figure(pooled,output/'figures')
        check(force=True);resources['status']='completed'
    except Exception as exc:
        _begin_cleanup();resources.update(status='stopped',stop_reason=type(exc).__name__,stop_detail=str(exc) if isinstance(exc,BudgetExceeded) else 'See preserved diagnostics; noautomaticretry')
        if allocations and not (output/'original_allocation_metrics.csv').exists():write_csv(output/'original_allocation_metrics.csv',allocations)
        if comparisons and not (output/'original_comparison_metrics.csv').exists():write_csv(output/'original_comparison_metrics.csv',comparisons)
        raise
    finally:
        _begin_cleanup();resources.update(completed_cells=len(completed),completed_cell_keys=completed,incomplete_cells=[list(c) for c in fixed_cells() if list(c) not in completed],
            completed_originals=len(comparisons)//3,observed_inner_streams=observed_streams,observed_logical_intervals=3*observed_streams,observed_invalid_logical_intervals=observed_invalid,
            bootstrap_samples=observed_streams*4999,logical_resampling_indices=observed_streams*4999*20,saved_audit_streams=len(list((output/'streams').glob('*.npz'))),finished_utc=datetime.now(timezone.utc).isoformat());_final_resources(output,resources)
    return resources


def run_smoke(output):
    inputs=load_inputs(make_inputs_manifest(root=ROOT),root=ROOT);output=output.resolve()
    if ROOT not in output.parents:raise ValueError('Smoke output must be inside project')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('Existing smoke attempt preserved')
    attempt=dict(question_id='RQ-M08',scope='smokeonly:savedouter0/R2/common12/B99 allocation4x3/2x6/1x12,noscientificdecisions',originals_per_cell=1,R=2,streams_per_budget=12,B=99,namespace=SMOKE_NAMESPACE,source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})
    _write_json(output/'attempt.json',attempt);result,completed={**attempt,'environment':runtime_environment(),'status':'started'},[]
    try:
        for dist,seed in fixed_cells():
            panel,tables=run_cell(dist,seed,inputs[(dist,seed)],O=1,R=2,streams=12,B=99,namespace=SMOKE_NAMESPACE,check=lambda:enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output)))
            completed.append(dict(distribution=dist,master_seed=seed,valid_entries=int(panel['valid'].sum())))
        result['status']='completed'
    except Exception as exc:result.update(status='stopped',stop_reason=type(exc).__name__);raise
    finally:
        _begin_cleanup();result.update(completed_cells=len(completed),cells=completed,mother_samples_generated=0,saved_originals_used=len(completed),inner_streams=24*len(completed),bootstrap_samples=2376*len(completed),logical_resampling_indices=47520*len(completed));_final_resources(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-manifest',action='store_true');modes.add_argument('--run',action='store_true');modes.add_argument('--smoke',action='store_true')
    parser.add_argument('--manifest',type=Path,default=ROOT/'evidence/rq_m08/manifest.json');parser.add_argument('--output',type=Path);parser.add_argument('--source-commit');args=parser.parse_args()
    if args.prepare_manifest:
        path=args.manifest.resolve()
        if ROOT not in path.parents:parser.error('Manifest must be inside project')
        path.parent.mkdir(parents=True,exist_ok=True);_write_json(path,make_manifest());print('Prepared',path.relative_to(ROOT))
    elif args.smoke:
        result=run_smoke(args.output or ROOT/'evidence/rq_m08/smoke');print(json.dumps({k:result[k] for k in ('status','completed_cells','process_wall_seconds','process_peak_rss_bytes')}))
    else:
        if not args.source_commit:parser.error('--run requires full reviewed --source-commit')
        result=run_benchmark(args.manifest.resolve(),args.output or ROOT/'evidence/rq_m08',args.source_commit);print(json.dumps({k:result[k] for k in ('status','completed_cells','process_wall_seconds','process_peak_rss_bytes')}))


def assemble_hypotheses(comparisons):
    index=_index(comparisons);quantities={};population_decisions={};seed_details=[]
    for dist in DISTRIBUTIONS:
        for name,field in (("H1","d"),("H2_L","L"),("H2_U","U")):
            values=_values(index,dist,(0,1,2),"bca",field);stats=mean_t_interval(values,confidence=.995)
            quantities[name+"_"+dist]={**stats,"field":field,"population":dist,"conditional_interval":mean_t_interval(values[np.isfinite(values)],confidence=.995)}
        first=quantities["H1_"+dist];first.update(band_status(first["low"],first["high"],complete=bool(first["valid"])))
        lower,upper=quantities["H2_L_"+dist],quantities["H2_U_"+dist]
        q=_values(index,dist,(0,1,2),"bca","Q30");mean_q=float(q.mean()) if np.isfinite(q).all() else np.nan
        population_decisions[dist]=dict(H1=dict(status=first["status"],reason=first["reason"]),H2={**variance_band_status(lower["low"],lower["high"],upper["low"],upper["high"],mean_q,complete=bool(lower["valid"] and upper["valid"])),"mean_Q30":mean_q,"lower_interval":"H2_L_"+dist,"upper_interval":"H2_U_"+dist})
    for seed in (0,1,2):seed_details.append(dict(master_seed=seed,**{f"{name}_{dist}":mean_t_interval(_expected(index,dist,seed,"bca",field),confidence=.995) for dist in DISTRIBUTIONS for name,field in (("H1","d"),("H2_L","L"),("H2_U","U"))}))
    return dict(question_id="RQ-M08",confirmatory_intervals=6,decision_confidence=.995,independent_originals_per_population=30,target_method="bca",B=4999,R=20,
        allocations=DESIGN["allocations"],primary_comparison="P120-P30",P60="descriptive only",seed_direction_gate=False,H1_margin=.0025,H1_units="Ase units,not percentagepoints",H2_variance_ratio_band=[.8,1.25],
        quantities=quantities,population_decisions=population_decisions,seed_details=seed_details,
        hypotheses={H:dict(status=joint_status([population_decisions[d][H]["status"] for d in DISTRIBUTIONS]),**{d:population_decisions[d][H]["status"] for d in DISTRIBUTIONS}) for H in ("H1","H2")})


def decomposition_summary(comparisons,decompositions):
    index=_index(comparisons);output=[]
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2,"pooled"):
            seeds=(0,1,2) if seed=="pooled" else (seed,)
            for method in METHODS:
                group=[r for r in decompositions if r["distribution"]==dist and r["method"]==method and r["master_seed"] in seeds]
                row=dict(distribution=dist,master_seed=seed,method=method,original_total=30 if seed=="pooled" else 10,diagnostic_budget_repeats=len(group),interpretation="r-level signs/quantiles descriptive; original-level intervals N10/30; no M07Rgap")
                for gap in GAPS:
                    raw=np.array([r[gap] for r in group],dtype=float);finite=raw[np.isfinite(raw)]
                    row.update({gap+"_finite_count":len(finite),gap+"_negative_count":int((finite<0).sum()),gap+"_zero_count":int((finite==0).sum()),gap+"_positive_count":int((finite>0).sum()),gap+"_raw_mean":float(finite.mean()) if len(finite) else np.nan})
                    for q,value in zip(QUANTILES,np.quantile(finite,QUANTILES,method="linear") if len(finite) else np.full(5,np.nan)):row[f"{gap}_raw_p{int(q*100):02d}"]=float(value)
                    values=_values(index,dist,seeds,method,"mean_"+gap)
                    for prefix,array in (("",values),("conditional_",values[np.isfinite(values)])):
                        for key,value in mean_t_interval(array,confidence=.995).items():row[f"{prefix}{gap}_original_{key}"]=value
                diagnostic_keys=sorted({key for r in group for key in r if "residual" in key and not key.endswith("_sign")})
                for field in diagnostic_keys:
                    raw=np.array([r[field] for r in group],dtype=float);finite=raw[np.isfinite(raw)];row[field+"_max_abs"]=float(np.max(np.abs(finite))) if len(finite) else np.nan
                row.update(algebra_failed_repeats=sum(r["reason"]=="algebraic_identity_failed" for r in group),algebra_unchecked_repeats=sum(not r["algebra_ok"] and r["reason"]!="algebraic_identity_failed" for r in group),algebra_passed_repeats=sum(bool(r["algebra_ok"]) for r in group))
                ratio=_values(index,dist,seeds,method,"ratio_Q120_Q30");q30=_values(index,dist,seeds,method,"Q30");q120=_values(index,dist,seeds,method,"Q120");finite=np.isfinite(ratio);paired=np.isfinite(q30)&np.isfinite(q120)
                row.update(ratio_planned_originals=len(ratio),ratio_finite_originals=int(finite.sum()),ratio_missing_or_undefined_originals=int((~finite).sum()),
                    mean_individual_Q120_Q30=float(ratio.mean()) if finite.all() else np.nan,conditional_mean_individual_Q120_Q30=float(ratio[finite].mean()) if finite.any() else np.nan,
                    ratio_mean_Q120_over_mean_Q30=float(q120.mean()/q30.mean()) if paired.all() and q30.mean()>0 else np.nan,
                    conditional_ratio_mean_Q120_over_mean_Q30=float(q120[paired].mean()/q30[paired].mean()) if paired.any() and q30[paired].mean()>0 else np.nan,
                    conditional_ratio_of_means_originals=int(paired.sum()),ratio_reference="descriptive; mean individual ratios differs from ratio meanQ; H2 uses meanL/U; zero denominatorNA")
                output.append(row)
    return output


if __name__=='__main__':main()
