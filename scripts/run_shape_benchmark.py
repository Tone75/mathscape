"""Preregister and run the one fixed RQ-M04 distribution-shape panel.

Main and smoke are explicit CLI operations, never ordinary pytest operations.
No previous RQ outputs or sources are changed. Results are never overwritten.
"""

from time import perf_counter
_PROCESS_START = perf_counter()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys


class BudgetExceeded(RuntimeError):
    pass


def _deadline(signum, frame):
    raise BudgetExceeded("300-second process wall budget reached")


if __name__ == "__main__" and ("--run" in sys.argv or "--smoke" in sys.argv):
    signal.signal(signal.SIGALRM, _deadline)
    signal.signal(signal.SIGTERM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 294-(perf_counter()-_PROCESS_START)))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathscape.distribution_shape import (
    generate_sample, generate_bootstrap_means, prepare_sample, infer_prefixes,
    original_metrics, mean_t_interval, welch_interval, quantity_status,
)
from scripts.run_bootstrap_benchmark import environment, sha256, peak_rss_bytes, directory_bytes, _write_json, write_csv
from scripts.run_stability_benchmark import sample_hash

DISTRIBUTIONS = ("normal", "gamma_0.5", "gamma_1", "gamma_2", "gamma_4", "gamma_16")
SHAPES = dict(zip(DISTRIBUTIONS[1:], [.5,1.,2.,4.,16.]))
METHODS = ("t", "percentile", "bca")
MAIN_NAMESPACE, SMOKE_NAMESPACE = 0x4D3034, 0x534D3034
DESIGN = dict(distributions=list(DISTRIBUTIONS), gamma_shapes=SHAPES, n=20,
              master_seeds=[0,1,2], R=400, K=10, B_values=[999,4999], alpha=.05,
              true_mean=0., true_variance=1., methods=list(METHODS))
POLICY = dict(wall_seconds=300, peak_rss_bytes=2*1024**3, output_bytes=50*1024**2,
              internal_stop_seconds=294, cleanup_reserve_seconds=6,
              max_resample_elements=1000000, confirmatory_quantities=2, decision_confidence=.99)
SOURCES = ("mathscape/distribution_shape.py", "mathscape/stability.py", "mathscape/bootstrap.py", "mathscape/inference.py",
           "scripts/run_shape_benchmark.py", "scripts/run_stability_benchmark.py", "scripts/run_bootstrap_benchmark.py",
           "scripts/audit_shape.py", "scripts/run_shape_guarded.py", "tests/test_shape_guard.py",
           "tests/test_distribution_shape.py", "tests/test_shape_runner.py", "tests/test_shape_audit.py")
FIELDS = ("lower","upper","z0","a","q_low","q_high","tau_low","tau_high","valid","reason","tail_warning")
METRICS = ("lower_sd","upper_sd","S","Ase","Apop","C","Mbelow","Mabove","T","mean_width","warning_rate")
METHOD_PAIRS = (("percentile","t"),("bca","percentile"),("bca","t"))
PAIR_METRICS = ("C","Ase","Apop","T","mean_width")


def fixed_cells():
    return [(distribution, seed) for distribution in DISTRIBUTIONS for seed in (0,1,2)]


def make_manifest():
    return {
        "schema_version":1,"question_id":"RQ-M04","created_utc":datetime.now(timezone.utc).isoformat(),
        "question":"With mean 0, variance 1 and n20 held fixed, how do standardized-gamma shapes differ in BCa coverage and miss direction, separately from finite-B endpoint variability?",
        "design":DESIGN,
        "totals":dict(generation_cells=18,original_samples=7200,inner_streams=72000,original_metric_rows=43200,
                      seed_summary_rows=108,pooled_summary_rows=36,bootstrap_samples=359928000,logical_resampling_indices=7198560000),
        "population_contract":{
            "gamma":"G~Gamma(shape=r,scale=1), X=(G-r)/sqrt(r), using population moments; never restandardize each original sample",
            "normal":"independent N(0,1) control, not identical to gamma r16",
            "shape_properties":"skewness=2/sqrt(r), excess kurtosis=6/r, lower bound=-sqrt(r); all change jointly, not an isolated causal skewness effect",
            "r1":"equal in distribution to Exp(scale=1)-1, but a new independently keyed panel; never reuse old RQ data",
        },
        "randomness":{
            "generator":"Generator(PCG64(SeedSequence(...)))","entropy":"master_seed",
            "main_namespace":MAIN_NAMESPACE,"smoke_namespace":SMOKE_NAMESPACE,
            "distribution_ids":dict(zip(DISTRIBUTIONS,range(1,7))),
            "spawn_key":["namespace","distribution_id","n","outer_id","purpose_id","inner_id"],
            "purpose_ids":dict(original_sample=0,bootstrap=1),"original_inner_id":0,
            "outer_ids":"0..399","bootstrap_inner_ids":"0..9",
            "excluded_from_key":["B","method","display_order","batch_size","alpha"],
            "prefix":"generate one length4999 bootstrap-mean vector per original/inner pair; use prefixes999/4999; recompute each prefix z0/q; cached t/a depend only on original",
            "independence":"different distributions are independent groups even with the same seed/outer ID; same original/B/inner shares means between percentile and BCa; K and B are not independent population sample counts",
        },
        "inference":{
            "definitions":"unchanged RQ-M02/M03: ddof1 t, linear empirical quantiles, half-tie z0, mean jackknife acceleration, prior BCa denominator/quantile validity rules",
            "truth_separation":"inference receives only original sample/bootstrap mean vector/alpha, never population/r/true mean or variance",
            "no_substitution":"no fallback, quantile clipping or adaptive B; invalid intervals retain reasons",
            "tau":"B*min(q,1-q) per endpoint; either tau<10 warns, without exclusion or invalidation",
        },
        "metrics":{
            "C":"mean_k I(lower<=0<=upper)","Mbelow":"mean_k I(upper<0), interval wholly below truth",
            "Mabove":"mean_k I(lower>0), interval wholly above truth","T":"Mbelow-Mabove",
            "partition":"C+Mbelow+Mabove=1 for complete valid originals",
            "S":"sqrt((sample Var_k(lower)+sample Var_k(upper))/2), ddof1",
            "Ase":"S/(sample SD ddof1/sqrt(20))","Apop":"S/(1/sqrt(20)), simulation evaluation only; never used to construct inference intervals",
            "t_control":"K/B copies are display references; actual interval count=R, S/Ase/Apop/endpoint SD=0 if valid",
            "aggregation":"within-original K means first, then independent originals; R400 per seed/N1200 pooled; no Wilson or sd/sqrt(R*K)",
            "intervals":"99% original-level mean t intervals, no clipping to rate range; zero variance gives undefined decision bounds with descriptive mean retained",
            "paired_B":"B4999-B999 within original, descriptive C/Ase/Apop/T/mean_width differences with original-level MCSE",
            "paired_methods":"percentile-t, bca-percentile, bca-t within original at each B; descriptive C/Ase/Apop/T/mean_width differences",
        },
        "hypotheses":{
            "H1":dict(method="bca",B=4999,quantity="mean(C,gamma_0.5)-mean(C,gamma_16)",
                      design="independent 2-group Welch 99% interval, each N1200; Satterthwaite df",
                      threshold=-.02,direction="less",support="pooled upper<-.02 and all 3 seed point differences<0",
                      not_supported="pooled lower>-.02",seed_threshold=0.),
            "H2":dict(method="bca",B=4999,distribution="gamma_0.5",quantity="mean(T)",
                      design="original-level t99% interval, N1200,df1199",threshold=.03,direction="greater",
                      support="pooled lower>.03 and all 3 seed point means>0",not_supported="pooled upper<.03",seed_threshold=0.),
            "status":"strict interval and separate seed conditions required; touching/straddling threshold or seed failure=>inconclusive. Invalid/unfinished needed originals or zero variance=>inconclusive. Not-supported is not zero-difference/symmetry proof",
            "multiplicity":"2 quantities with99% intervals, approximate multiplicity control, not exact simultaneous guarantee",
            "descriptive":"other distributions, B999, endpoint stability, warnings, method/B differences remain descriptive; no promotion based on significance, no monotonicity/causal/limit-B claims",
        },
        "invalid_policy":{
            "original":"any required invalid inner interval makes that method/B original C/T/S/Ase/Apop/width/warning primary metrics NA",
            "comparison":"missing/invalid original needed by H1/H2 makes that main quantity NA/inconclusive; unrelated populations or B999 invalids do not poison complete confirmatory targets",
            "conditional":"jointly complete original references use conditional_ prefix and explicit denominator; never invalid-as-noncoverage or silent omission",
            "budgets":"retain planned/observed/complete original counts and actual valid/invalid interval counts; stop preserving complete/incomplete cells",
        },
        "storage":{
            "panel_axes":["original","B","method","inner"],"original_hash":"SHA256 of little-endian float64 C-order vector bytes",
            "saved":"originals,IDs/hashes,endpoints,z0/a/q/tau,valid/reason/warnings; no full bootstrap/index arrays",
            "capacity":"float64 numeric endpoints/diagnostics, compressed NPZ, compact reason strings; no retention of all original bootstrap arrays",
            "audit_limit":"saved diagnostics cannot prove actual bootstrap-prefix generation or raw rank counts; frozen source and deterministic references cover that contract",
        },
        "policy":POLICY,
        "execution":{
            "main":"once only, no added seeds/R/K/B or redraw on unsupported/inconclusive/invalid/timeout",
            "outer_deadline":"standard entry is scripts/run_shape_guarded.py; parent watchdog controls the full worker CLI with TERM at295s, KILL at298s and finish target299s. Worker soft-stops at294s with cleanup reserve, and ignores repeated TERM during final resource writes",
            "smoke":"one separately authorized 18cell/R2/K4/prefix19,99 check in smoke namespace;36 originals,144 inner streams,14256 bootstrap samples; no scientific selection",
            "bug_correction":"actual code bug only, at most one separately authorized attempt retaining old source/results/reason, cumulative mainwall<=600s",
            "stop":"preserve complete/incomplete cells without automatic shrinking/expansion/retry",
        },
        "environment":environment(),"source_files_sha256":{name:sha256(ROOT/name) for name in SOURCES},
    }


def validate_manifest(manifest):
    if manifest.get("schema_version")!=1 or manifest.get("question_id")!="RQ-M04" or manifest.get("design")!=DESIGN or manifest.get("policy")!=POLICY:
        raise ValueError("Manifest design/policy differs from fixed RQ-M04")


def verify_committed_inputs(path,manifest,commit):
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if actual!=commit or len(commit)!=40:
        raise ValueError("--source-commit must be full current reviewed HEAD")
    for name in [*SOURCES,str(path.resolve().relative_to(ROOT))]:
        if subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=ROOT)!=(ROOT/name).read_bytes():
            raise ValueError(f"Committed input differs: {name}")
    for name in SOURCES:
        if sha256(ROOT/name)!=manifest["source_files_sha256"].get(name):
            raise ValueError(f"Preregistered source differs: {name}")
    if environment()!=manifest["environment"]:
        raise ValueError("Runtime environment differs from preregistration")


def enforce_budget(wall,rss,output):
    if wall>=294 or rss>=2*1024**3 or output+65536>=50*1024**2:
        raise BudgetExceeded("wall/RSS/output budget reached; preserving partial results")


def _allocate(R,Bs,K):
    shape=(R,len(Bs),3,K)
    return {field:(np.full(shape,"not_computed",dtype="U64") if field=="reason" else
                   np.zeros(shape,bool) if field in ("valid","tail_warning") else np.full(shape,np.nan)) for field in FIELDS}


def run_cell(distribution,seed,R=400,K=10,Bs=(999,4999),namespace=MAIN_NAMESPACE,
             check=lambda:None,sample_fn=None,bootstrap_fn=None):
    if (not isinstance(R,int) or isinstance(R,bool) or not 1<=R<=400 or
            not isinstance(K,int) or isinstance(K,bool) or not 2<=K<=10):
        raise ValueError("R/K must be integers in1..400/2..10")
    if len(Bs)!=2 or any(not isinstance(b,int) or isinstance(b,bool) or not 1<=b<=4999 for b in Bs) or list(Bs)!=sorted(set(Bs)):
        raise ValueError("Bs must be two increasing integer prefixes<=4999")
    sample_fn,bootstrap_fn=sample_fn or generate_sample,bootstrap_fn or generate_bootstrap_means
    data=_allocate(R,Bs,K)
    samples,hashes,rows=np.empty((R,20)),[],[]
    for outer in range(R):
        check()
        sample=sample_fn(distribution,20,master_seed=seed,outer_id=outer,namespace=namespace)
        samples[outer]=sample;hashes.append(sample_hash(sample))
        context=prepare_sample(sample,alpha=.05);se=float(context["se"])
        for inner in range(K):
            check();failure=""
            if not context["valid"]:
                means=np.full(max(Bs),np.nan)
            else:
                try:
                    means=bootstrap_fn(sample,distribution,20,master_seed=seed,outer_id=outer,inner_id=inner,
                                       Bmax=max(Bs),namespace=namespace,max_elements=1000000)
                except ValueError as exc:
                    if str(exc)!="bootstrap_mean_nonfinite":raise
                    failure=str(exc);means=np.full(max(Bs),np.nan)
            records=infer_prefixes(context,means,B_values=Bs)
            for bi,B in enumerate(Bs):
                for mi,method in enumerate(METHODS):
                    if failure and method!="t":records[B][method]["reason"]=failure
                    for field in FIELDS:data[field][outer,bi,mi,inner]=records[B][method][field]
        metrics=original_metrics({field:array[outer] for field,array in data.items()},se,n=20)
        for bi,B in enumerate(Bs):
            for mi,method in enumerate(METHODS):
                row=dict(distribution=distribution,n=20,master_seed=seed,outer_id=outer,B=B,method=method,
                         R=R,K=K,alpha=.05,se=se,sample_sha256=hashes[-1])
                row.update({key:value[bi,mi].item() for key,value in metrics.items()})
                rows.append(row)
    check()
    return data,samples,np.array(hashes),rows


def summarize_rows(originals,pooled=False):
    output=[]
    for distribution in DISTRIBUTIONS:
        for seed in (["pooled"] if pooled else [0,1,2]):
            for B in (999,4999):
                for method in METHODS:
                    group=[r for r in originals if r["distribution"]==distribution and r["B"]==B and r["method"]==method and (pooled or r["master_seed"]==seed)]
                    expected=1200 if pooled else 400
                    complete=[r for r in group if r["complete"]]
                    row=dict(distribution=distribution,n=20,master_seed=seed,seed_group="0,1,2" if pooled else str(seed),
                             B=B,method=method,R=expected,K=10,alpha=.05,original_total=expected,
                             observed_original_count=len(group),missing_original_count=expected-len(group),
                             complete_original_count=len(complete),incomplete_original_count=expected-len(complete),
                             interval_count=sum(r["total_inner"] for r in group),valid_count=sum(r["valid_inner"] for r in group),
                             invalid_count=sum(r["invalid_inner"] for r in group),conditional_original_count=len(complete))
                    for metric in METRICS:
                        main_values=[r[metric] if r["complete"] else float("nan") for r in group]
                        if len(group)!=expected:main_values += [float("nan")]*max(1,expected-len(group))
                        for prefix,values in (("",main_values),("conditional_",[r[metric] for r in complete])):
                            stats=mean_t_interval(values,confidence=.99)
                            for suffix,key in (("","mean"),("_sd","sd"),("_mcse","mcse"),("_decision_low","low"),("_decision_high","high")):
                                row[f"{prefix}{metric}{suffix}"]=stats[key]
                    output.append(row)
    return output


def _index_originals(originals):
    index={}
    for row in originals:
        key=(row["distribution"],row["master_seed"],row["B"],row["method"])
        group=index.setdefault(key,{})
        if row["outer_id"] in group or row["outer_id"] not in range(400):
            raise ValueError("Duplicate/out-of-range original IDs")
        group[row["outer_id"]]=row
    return index


def _expected(originals,distribution,seed,B,method,metric):
    index=originals if isinstance(originals,dict) else _index_originals(originals)
    by_id=index.get((distribution,seed,B,method),{})
    return np.array([by_id[i][metric] if i in by_id and by_id[i]["complete"] else float("nan") for i in range(400)])


def assemble_hypotheses(originals):
    originals=_index_originals(originals)
    group1=[_expected(originals,"gamma_0.5",seed,4999,"bca","C") for seed in (0,1,2)]
    group0=[_expected(originals,"gamma_16",seed,4999,"bca","C") for seed in (0,1,2)]
    x1,x0=np.concatenate(group1),np.concatenate(group0)
    h1=welch_interval(x1,x0,confidence=.99)
    h1_seeds=[float(np.mean(a)-np.mean(b)) for a,b in zip(group1,group0)]
    h1_status=quantity_status(h1["low"],h1["high"],-.02,"less",h1_seeds,seed_threshold=0.)
    h1.update(status=h1_status,threshold=-.02,seed_threshold=0.,direction="less",seed_points=h1_seeds,
              group1="gamma_0.5",group0="gamma_16",independent_groups=True,
              conditional_welch=welch_interval(x1[np.isfinite(x1)],x0[np.isfinite(x0)],confidence=.99))
    h2_seeds=[_expected(originals,"gamma_0.5",seed,4999,"bca","T") for seed in (0,1,2)]
    x2=np.concatenate(h2_seeds);h2=mean_t_interval(x2,confidence=.99)
    points=[float(np.mean(values)) for values in h2_seeds]
    h2.update(status=quantity_status(h2["low"],h2["high"],.03,"greater",points,seed_threshold=0.),
              threshold=.03,seed_threshold=0.,direction="greater",seed_points=points,
              conditional_interval=mean_t_interval(x2[np.isfinite(x2)],confidence=.99))
    seed_details=[]
    for seed,a,b,t in zip((0,1,2),group1,group0,h2_seeds):
        seed_details.append(dict(master_seed=seed,H1=welch_interval(a,b,confidence=.99),H2=mean_t_interval(t,confidence=.99)))
    return dict(question_id="RQ-M04",confirmatory_quantities=2,decision_confidence=.99,
                independent_originals_per_population=1200,target_method="bca",target_B=4999,
                quantities=dict(H1=h1,H2=h2),seed_details=seed_details,
                hypotheses=dict(H1=dict(status=h1_status),H2=dict(status=h2["status"])))


def descriptive_pairs(originals,kind):
    originals=_index_originals(originals)
    output=[]
    for distribution in DISTRIBUTIONS:
        for seed in (0,1,2,"pooled"):
            seeds=(0,1,2) if seed=="pooled" else (seed,)
            comparisons=[(method,method,4999,999) for method in METHODS] if kind=="B" else [(m,r,B,B) for B in (999,4999) for m,r in METHOD_PAIRS]
            for method,reference,B,reference_B in comparisons:
                row=dict(distribution=distribution,n=20,master_seed=seed,method=method,reference=reference,
                         B=B,reference_B=reference_B,original_total=1200 if seed=="pooled" else 400,
                         interpretation="descriptive paired original difference, not equivalence or convergence")
                for metric in PAIR_METRICS:
                    differences=np.concatenate([_expected(originals,distribution,s,B,method,metric)-_expected(originals,distribution,s,reference_B,reference,metric) for s in seeds])
                    stats=mean_t_interval(differences,confidence=.99)
                    conditional=mean_t_interval(differences[np.isfinite(differences)],confidence=.99)
                    row[f"{metric}_complete_original_count"]=stats["finite_count"]
                    for prefix,result in (("",stats),("conditional_",conditional)):
                        for suffix,key in (("_difference","mean"),("_sd","sd"),("_mcse","mcse"),("_decision_low","low"),("_decision_high","high")):
                            row[f"{prefix}{metric}{suffix}"]=result[key]
                output.append(row)
    return output


def write_figure(pooled,output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    output.mkdir(exist_ok=False)
    fig=make_subplots(rows=2,cols=2,subplot_titles=["BCa coverage: population shape","BCa miss direction (B4999)","BCa endpoint variability: Ase","BCa endpoint variability: Apop"])
    for B,dash in ((999,"dash"),(4999,"solid")):
        rows=[next(r for r in pooled if r["distribution"]==d and r["B"]==B and r["method"]=="bca") for d in DISTRIBUTIONS]
        for row,col,metric in ((1,1,"C"),(2,1,"Ase"),(2,2,"Apop")):
            fig.add_trace(go.Scatter(x=list(DISTRIBUTIONS),y=[r[metric] for r in rows],mode="lines+markers",name=f"B={B}",legendgroup=str(B),showlegend=row==col==1,
                                    line=dict(dash=dash),error_y=dict(type="data",symmetric=False,array=[r[f"{metric}_decision_high"]-r[metric] for r in rows],arrayminus=[r[metric]-r[f"{metric}_decision_low"] for r in rows])),row=row,col=col)
        if B==4999:
            for metric in ("Mbelow","Mabove"):
                fig.add_trace(go.Scatter(x=list(DISTRIBUTIONS),y=[r[metric] for r in rows],mode="lines+markers",name=metric),row=1,col=2)
    fig.add_hline(y=.95,line_dash="dot",row=1,col=1)
    fig.update_xaxes(type="category",title_text="Independent population shapes; normal is a separate control")
    fig.update_layout(template="plotly_white",width=1400,height=950,
                      title="RQ-M04: 1200 original samples per population; bars=99% original-level MC intervals; K=10",margin=dict(t=100,b=130),legend=dict(orientation="h",y=-.2))
    fig.write_html(str(output/"overview.html"),include_plotlyjs=True,full_html=True,config={"displaylogo":False})


def _final_resources(output,record):
    record.update(process_wall_seconds=perf_counter()-_PROCESS_START,process_peak_rss_bytes=peak_rss_bytes(),output_bytes_before_resources=directory_bytes(output))
    record["output_bytes_total"]=record["output_bytes_before_resources"]
    for _ in range(5):
        total=record["output_bytes_before_resources"]+len((json.dumps(record,indent=2,allow_nan=False)+"\n").encode())
        if total==record["output_bytes_total"]:break
        record["output_bytes_total"]=total
    _write_json(output/"resources.json",record)


def run_benchmark(manifest_path,output,source_commit):
    manifest=json.loads(manifest_path.read_text());validate_manifest(manifest)
    verify_committed_inputs(manifest_path,manifest,source_commit)
    output=output.resolve()
    if ROOT not in output.parents:raise ValueError("Output must be inside project")
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ("attempt.json","resources.json","panels","summary.csv","original_metrics.csv","figures")):
        raise FileExistsError("Existing attempt/results preserved")
    attempt=dict(question_id="RQ-M04",status="started",started_utc=datetime.now(timezone.utc).isoformat(),source_git_sha=source_commit,manifest_sha256=sha256(manifest_path),namespace=MAIN_NAMESPACE,single_main_attempt=True)
    _write_json(output/"attempt.json",attempt);(output/"panels").mkdir()
    originals,completed=[],[]
    resources={**attempt,"environment":environment(),"R":400,"K":10,"B_values":[999,4999],"n":20,"total_cells":18,
               "bootstrap_arrays_saved":False,"original_samples_saved":True,"max_resample_elements":1000000,
               "wall_scope":"module start before heavy imports through calculation/output, excluding interpreter startup and final resources write; external time records full CLI"}
    def check():enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output))
    try:
        for distribution,seed in fixed_cells():
            resources["active_cell"]=[distribution,seed]
            data,samples,hashes,rows=run_cell(distribution,seed,check=check)
            with (output/"panels"/f"{distribution}_seed{seed}.npz").open("xb") as handle:
                np.savez_compressed(handle,**data,samples=samples,sample_sha256=hashes,outer_id=np.arange(400),B_values=np.array([999,4999]),methods=np.array(METHODS),inner_id=np.arange(10),distribution=distribution,n=20,master_seed=seed,R=400,K=10,alpha=.05,namespace=MAIN_NAMESPACE)
            originals.extend(rows);completed.append([distribution,seed]);check()
        write_csv(output/"original_metrics.csv",originals)
        summary,pooled=summarize_rows(originals),summarize_rows(originals,pooled=True)
        write_csv(output/"summary.csv",summary);write_csv(output/"pooled_summary.csv",pooled)
        write_csv(output/"paired_B_summary.csv",descriptive_pairs(originals,"B"))
        write_csv(output/"paired_method_summary.csv",descriptive_pairs(originals,"method"))
        _write_json(output/"hypotheses.json",assemble_hypotheses(originals));write_figure(pooled,output/"figures")
        check();resources["status"]="completed"
    except Exception as exc:
        resources.update(status="stopped",stop_reason=f"{type(exc).__name__}: {exc}")
        if originals and not (output/"original_metrics.csv").exists():write_csv(output/"original_metrics.csv",originals)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        resources.update(completed_cells=len(completed),completed_cell_keys=completed,incomplete_cells=[list(c) for c in fixed_cells() if list(c) not in completed],finished_utc=datetime.now(timezone.utc).isoformat())
        _final_resources(output,resources)
    return resources


def run_smoke(output):
    output=output.resolve()
    if ROOT not in output.parents:raise ValueError("Smoke output must be inside project")
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError("Existing smoke attempt preserved")
    attempt=dict(question_id="RQ-M04",scope="smoke only; no scientific decisions",R=2,K=4,B_values=[19,99],namespace=SMOKE_NAMESPACE,
                 source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})
    _write_json(output/"attempt.json",attempt)
    result,completed={**attempt,"environment":environment(),"status":"started"},[]
    try:
        for distribution,seed in fixed_cells():
            data,samples,hashes,rows=run_cell(distribution,seed,R=2,K=4,Bs=(19,99),namespace=SMOKE_NAMESPACE,
                                             check=lambda:enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output)))
            completed.append(dict(distribution=distribution,master_seed=seed,valid_stored_entries=int(data["valid"].sum())))
        result["status"]="completed"
    except Exception as exc:
        result.update(status="stopped",stop_reason=f"{type(exc).__name__}: {exc}");raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        result.update(completed_cells=len(completed),cells=completed,original_samples=2*len(completed),inner_streams=8*len(completed))
        _final_resources(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-manifest",action="store_true");modes.add_argument("--run",action="store_true");modes.add_argument("--smoke",action="store_true")
    parser.add_argument("--manifest",type=Path,default=ROOT/"evidence/rq_m04/manifest.json");parser.add_argument("--output",type=Path);parser.add_argument("--source-commit")
    args=parser.parse_args()
    if args.prepare_manifest:
        path=args.manifest.resolve()
        if ROOT not in path.parents:parser.error("Manifest must be inside project")
        path.parent.mkdir(parents=True,exist_ok=True);_write_json(path,make_manifest());print("Prepared",path.relative_to(ROOT))
    elif args.smoke:
        result=run_smoke(args.output or ROOT/"evidence/rq_m04/smoke")
        print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))
    else:
        if not args.source_commit:parser.error("--run requires full reviewed --source-commit")
        result=run_benchmark(args.manifest.resolve(),args.output or ROOT/"evidence/rq_m04",args.source_commit)
        print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))


if __name__=="__main__":main()
