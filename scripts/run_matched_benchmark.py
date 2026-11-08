"""Preregister/run the sole RQ-M05 matched-moments experiment.

Use run_matched_guarded.py for the main CLI. Samples/diagnostics are compact
NPZ; per-original records stream to CSV.GZ. Neither pytest nor UI runs main.
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
    raise BudgetExceeded("294-second internal soft stop; cleanup reserve before outer deadline")


if __name__=="__main__" and ("--run" in sys.argv or "--smoke" in sys.argv):
    signal.signal(signal.SIGALRM,_deadline);signal.signal(signal.SIGTERM,_deadline)
    signal.setitimer(signal.ITIMER_REAL,max(.001,294-(perf_counter()-_PROCESS_START)))

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from mathscape.matched_moments import (
    generate_sample,generate_bootstrap_means,prepare_sample,infer_single,original_metrics,
    mean_t_interval,welch_interval,practical_difference_status,LOGNORMAL_CONSTANTS,validate_population_moments,
)
from scripts.run_bootstrap_benchmark import environment,sha256,peak_rss_bytes,directory_bytes,_write_json,write_csv
from scripts.run_stability_benchmark import sample_hash

DISTRIBUTIONS=("gamma","lognormal")
METHODS=("t","percentile","bca")
MAIN_NAMESPACE,SMOKE_NAMESPACE=0x4D3035,0x534D3035
DESIGN=dict(distributions=list(DISTRIBUTIONS),n=20,master_seeds=[0,1,2],R=10000,K=1,B=4999,alpha=.05,
            true_mean=0.,true_variance=1.,true_skewness=2.,methods=list(METHODS))
POLICY=dict(wall_seconds=300,internal_stop_seconds=294,cleanup_reserve_seconds=6,
            peak_rss_bytes=2*1024**3,output_bytes=50*1024**2,max_resample_elements=1000000,
            confirmatory_quantities=2,decision_confidence=.99,practical_margin=.01)
SOURCES=("mathscape/matched_moments.py","mathscape/stability.py","mathscape/bootstrap.py","mathscape/inference.py",
         "scripts/run_matched_benchmark.py","scripts/run_matched_guarded.py","scripts/audit_matched.py",
         "scripts/run_bootstrap_benchmark.py","scripts/run_stability_benchmark.py",
         "tests/test_matched_moments.py","tests/test_matched_runner.py","tests/test_matched_guard.py","tests/test_matched_audit.py",
         "requirements.txt","evidence/rq_m05/SPEC.md","evidence/rq_m05/population_checks.json")
DIAGNOSTICS=("lower","upper","z0","a","q_low","q_high","tau_low","tau_high")
STABILITY=("lower_sd","upper_sd","S","Ase","Apop")
METRICS=("C","Mbelow","Mabove","T","mean_width","warning_rate",*STABILITY)
PAIR_METRICS=("C","T","mean_width")
METHOD_PAIRS=(("percentile","t"),("bca","percentile"),("bca","t"))
CSV_FIELDS=("distribution","master_seed","outer_id","method","complete","reason_code","reason",
            "valid_inner","total_inner","invalid_inner",*METRICS)


def runtime_environment():
    return {**environment(),"streamlit":version("streamlit"),"pytest":version("pytest")}


def fixed_cells():
    return [(dist,seed) for dist in DISTRIBUTIONS for seed in (0,1,2)]


def make_manifest():
    return dict(
        schema_version=1,question_id="RQ-M05",created_utc=datetime.now(timezone.utc).isoformat(),design=DESIGN,
        totals=dict(generation_cells=6,original_samples=60000,original_observations=1200000,inner_streams=60000,
                    bootstrap_samples=299940000,logical_resampling_indices=5998800000,original_method_records=180000,seed_summary_rows=18,pooled_summary_rows=6),
        populations=dict(
            gamma="G~Gamma(shape=1,scale=1),X=G-1; mean0 variance1 skewness2 excesskurtosis6 support[-1,infinity)",
            lognormal="v=2*sinh(asinh(1)/3),t=1+v^2,s=sqrt(log1p(v^2)),m=sqrt(t),d=sqrt(t)*v; X=(exp(sZ)-m)/d; v^3+3v=2; no rounded generation constants",
            lognormal_constants=LOGNORMAL_CONSTANTS,
            contract="theoretical standardization only, never force sample moments; infer arithmetic mean of X, not log mean; kurtosis/tails/support still differ, no isolated causal kurtosis claim",
            moment_checks="deterministic public SciPy mvsk check for mean0/variance1/skewness2 absolute error<=1e-10, and matching excess kurtosis/support"),
        randomness=dict(generator="Generator(PCG64(SeedSequence(...)))",entropy="master_seed",main_namespace=MAIN_NAMESPACE,smoke_namespace=SMOKE_NAMESPACE,
                        distribution_ids=dict(gamma=1,lognormal=2),spawn_key=["namespace","distribution_id","n","outer_id","purpose_id","inner_id"],
                        purpose_ids=dict(original_sample=0,bootstrap=1),inner_id=0,outer_ids="0..9999",
                        excluded_from_key=["B","method","UI_state","batch_size"],
                        independence="independent populations, never pair L/G by matching IDs; same-original percentile/BCa share one B4999 mean vector; originals plus their independent inner randomness are the population-repetition units",
                        no_prefix="one B level; no prefix comparison or old-panel reuse"),
        inference=dict(definitions="unchanged ddof1 t, linear quantiles, half-tie z0, jackknife acceleration and prior BCa validity rules",
                       truth_separation="infer_single accepts original context/bootstrap means only; no population label/parameters/truth",
                       invalid="reason-labelled, no fallback/clipping/adaptive B; warnings tau=B*min(q,1-q)<10 retained without exclusion"),
        metrics=dict(C="I(lower<=0<=upper)",Mbelow="I(upper<0), interval wholly below truth",Mabove="I(lower>0), interval wholly above truth",T="Mbelow-Mabove",
                     identity="C+Mbelow+Mabove=1 for valid intervals",mean_width="upper-lower",warning_rate="tail warning indicator",
                     independent_unit="original sample and its one independent bootstrap computation; R10000 per seed/N30000 per population; no sd/sqrt(N*B)",
                     stability="K1: lower_sd/upper_sd/S/Ase/Apop are NA for ALL methods including t; do not fill zero",
                     intervals="99% original-level mean t intervals for descriptive summaries; no clipping outside rate bounds; zero variance makes decision bounds undefined"),
        confirmation=dict(target="BCa B4999 only",sign="lognormal minus gamma, independent groups",
                          H1="difference in C",H2="difference in T",design="independent Welch99% intervals, ddof1/Satterthwaite, each N30000",
                          rule="lo>.01 or hi<-.01: supported practical difference (direction); lo>-.01 and hi<.01: not_supported practical-difference hypothesis AND equivalence_supported for this metric/these fixed conditions; otherwise touching/straddling=>inconclusive",
                          invalid="any required missing/invalid original, nonfinite or SE0=>NA/inconclusive; conditional complete-case references never decide",
                          seeds="all 3 seed differences/99% descriptive intervals reported; NO seed direction gate",
                          scope="both metrics must individually support equivalence before saying both within1pp; no general moment sufficiency/causal/infinite-B claim",
                          multiplicity="2 approximate99% intervals; Bonferroni sum noncoverage about<=2%, not exact finite-sample guarantee"),
        descriptive=dict(population_contrasts="all3methods C/T/width independent L-G Welch, allseed and pooled",
                         method_contrasts="all3 same-original method pairs C/T/width with paired original-level MCSE, allseed and pooled; descriptive only"),
        invalid_policy=dict(denominators="retain planned/observed/valid/invalid/missing counts; invalid never becomes noncoverage",
                            local_gate="other-method invalids do not poison complete BCa target; required invalid/missing gives mainNA",
                            conditional="conditional_ references with finite-count denominators only"),
        storage=dict(panel_axes=["original","method"],sample_dtype="float64 little-endian C-order",sample_sha256_dtype="S64 ASCII hex",
                     sample_hash="SHA256 of original little-endian float64 C-order bytes",outer_id_dtype="uint16",reason="uint8 codes indexing per-panel Unicode reason_labels, code0 empty-valid reason",
                     numeric_diagnostics="float64 lower/upper/z0/a/q_low/q_high/tau_low/tau_high, arrays(R,3)",masks="bool valid/tail_warning arrays(R,3)",
                     original_records="original_metrics.csv.gz, streaming 180000 rows, gzip mtime0; IDs join NPZ hashes, no repeated bootstrap arrays",
                     record_columns=list(CSV_FIELDS),full_bootstrap_saved=False,
                     estimate=dict(raw_sample_bytes=9600000,raw_numeric_diagnostic_bytes=11520000,raw_hash_bytes=3840000,raw_mask_reason_id_bytes=660000,
                                   npz_reserve_bytes=27*1024**2,csv_gz_reserve_bytes=10*1024**2,single_html_reserve_bytes=5*1024**2,other_public_reserve_bytes=6*1024**2,total_public_reserve_bytes=48*1024**2),
                     estimate_caveat="conservative allocation, not guaranteed compression; actual size guard stops without scientific resizing if unexpectedly exceeded",
                     audit_limit="saved samples/endpoints permit aggregate audit, not a complete reconstruction of all60000 bootstrap generation sequences; frozen code/fixtures cover RNG/resampling logic"),
        policy=POLICY,
        execution=dict(main="once only, no extra seeds/R/K/B or redraw on outcome/invalid/timeout",smoke="sole separately authorized6cells,R2,K1,B99 in smoke namespace;12originals/12streams/1188resamples; no scientific selection",
                       outer_guard="run_matched_guarded.py starts one child; TERM295s/KILL298s/finish target299s; worker soft-stop294s and reserve cleanup, outer CLI externally timed",
                       correction="actual code bug only, at most one separately authorized attempt retaining old source/results/reason, cumulative mainwall<=600s",
                       stop="save completed panels plus observed partial panel/CSV records and diagnostics, no automatic scale changes"),
        environment=runtime_environment(),source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})


def validate_manifest(m):
    if m.get("schema_version")!=1 or m.get("question_id")!="RQ-M05" or m.get("design")!=DESIGN or m.get("policy")!=POLICY:
        raise ValueError("Manifest differs from fixed RQ-M05")


def verify_committed_inputs(path,m,commit):
    if len(commit)!=40 or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=commit:
        raise ValueError("--source-commit must be full current reviewed HEAD")
    for name in [*SOURCES,str(path.resolve().relative_to(ROOT))]:
        if subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=ROOT)!=(ROOT/name).read_bytes():raise ValueError(f"Committed source differs: {name}")
    for name in SOURCES:
        if sha256(ROOT/name)!=m["source_files_sha256"].get(name):raise ValueError(f"Preregistered source differs: {name}")
    if runtime_environment()!=m["environment"]:raise ValueError("Runtime environment differs")
    validate_population_moments(tolerance=1e-10)


def enforce_budget(wall,rss,output):
    if wall>=294 or rss>=2*1024**3 or output+65536>=50*1024**2:raise BudgetExceeded("wall/RSS/output limit reached")


def _allocate(R):
    return {**{f:np.full((R,3),np.nan) for f in DIAGNOSTICS},"valid":np.zeros((R,3),bool),"tail_warning":np.zeros((R,3),bool),
            "reason_codes":np.zeros((R,3),np.uint8),"samples":np.empty((R,20)),"sample_sha256":np.empty(R,dtype="S64"),"outer_id":np.arange(R,dtype=np.uint16)}


def run_cell(distribution,seed,R=10000,B=4999,namespace=MAIN_NAMESPACE,check=lambda:None,sample_fn=None,bootstrap_fn=None,progress=None):
    if not isinstance(R,int) or isinstance(R,bool) or not 1<=R<=10000 or not isinstance(B,int) or isinstance(B,bool) or not 1<=B<=4999:raise ValueError("R/B must be integer1..10000/1..4999")
    sample_fn,bootstrap_fn=sample_fn or generate_sample,bootstrap_fn or generate_bootstrap_means
    data=_allocate(R);labels=[""];progress=progress if progress is not None else {}
    progress.update(panel=data,reason_labels=labels,observed_original_count=0)
    for outer in range(R):
        check();sample=sample_fn(distribution,20,master_seed=seed,outer_id=outer,namespace=namespace)
        data["samples"][outer]=sample;data["sample_sha256"][outer]=sample_hash(sample).encode("ascii")
        context=prepare_sample(sample,alpha=.05);failure=""
        if not context["valid"]:means=np.full(B,np.nan)
        else:
            try:means=bootstrap_fn(sample,distribution,20,master_seed=seed,outer_id=outer,inner_id=0,Bmax=B,namespace=namespace,max_elements=1000000)
            except ValueError as exc:
                if str(exc)!="bootstrap_mean_nonfinite":raise
                means=np.full(B,np.nan);failure=str(exc)
        records=infer_single(context,means)
        for mi,method in enumerate(METHODS):
            record=records[method]
            for field in DIAGNOSTICS:data[field][outer,mi]=record[field]
            data["valid"][outer,mi]=record["valid"];data["tail_warning"][outer,mi]=record["tail_warning"]
            reason=failure if failure and method!="t" else record["reason"]
            if reason not in labels:
                if len(labels)>=256 or len(reason)>96:raise ValueError("Reason dictionary capacity exceeded")
                labels.append(reason)
            data["reason_codes"][outer,mi]=labels.index(reason)
        progress["observed_original_count"]=outer+1
    check();data["reason_labels"]=np.array(labels,dtype="U96")
    return data


def panel_metrics(panel):
    return {**original_metrics({key:panel[key] for key in ("lower","upper","valid","tail_warning")}),"outer_id":panel["outer_id"]}


def expected_values(cells,distribution,seed,method,metric):
    values=np.full(10000,np.nan);cell=cells.get((distribution,seed))
    if cell is None:return values
    ids=np.asarray(cell["outer_id"]);mi=METHODS.index(method)
    if len(set(ids.tolist()))!=len(ids) or np.any((ids<0)|(ids>=10000)):raise ValueError("Duplicate/out-of-range original IDs")
    values[ids]=np.where(cell["complete"][:,mi],cell[metric][:,mi],np.nan)
    return values


def _values(cells,dist,seeds,method,metric):
    return np.concatenate([expected_values(cells,dist,seed,method,metric) for seed in seeds])


def summarize(cells,pooled=False):
    rows=[]
    for dist in DISTRIBUTIONS:
        for seed in (["pooled"] if pooled else [0,1,2]):
            seeds=(0,1,2) if pooled else (seed,);expected=30000 if pooled else 10000
            for mi,method in enumerate(METHODS):
                observed=sum(len(cells[(dist,s)]["outer_id"]) for s in seeds if (dist,s) in cells)
                valid=sum(int(cells[(dist,s)]["complete"][:,mi].sum()) for s in seeds if (dist,s) in cells)
                row=dict(distribution=dist,master_seed=seed,n=20,R=expected,K=1,B=4999,alpha=.05,method=method,
                         original_total=expected,observed_original_count=observed,missing_original_count=expected-observed,
                         valid_count=valid,invalid_count=observed-valid,complete_original_count=valid,
                         incomplete_original_count=expected-valid,
                         interval_count=observed,conditional_original_count=valid,stability_applicable=False,stability_reason="K1: not estimated for any method")
                for metric in METRICS:
                    values=_values(cells,dist,seeds,method,metric)
                    for prefix,array in (("",values),("conditional_",values[np.isfinite(values)])):
                        stats=mean_t_interval(array,confidence=.99)
                        for suffix,key in (("","mean"),("_sd","sd"),("_mcse","mcse"),("_decision_low","low"),("_decision_high","high")):
                            row[f"{prefix}{metric}{suffix}"]=stats[key]
                rows.append(row)
    return rows


def population_contrasts(cells):
    rows=[]
    for seed in (0,1,2,"pooled"):
        seeds=(0,1,2) if seed=="pooled" else (seed,)
        for method in METHODS:
            row=dict(master_seed=seed,method=method,n=20,K=1,B=4999,alpha=.05,group1="lognormal",group0="gamma",independent_groups=True)
            for metric in PAIR_METRICS:
                one,zero=(_values(cells,dist,seeds,method,metric) for dist in ("lognormal","gamma"))
                for prefix,a,b in (("",one,zero),("conditional_",one[np.isfinite(one)],zero[np.isfinite(zero)])):
                    stats=welch_interval(a,b,confidence=.99)
                    for key,value in stats.items():row[f"{prefix}{metric}_{key}"]=value
            rows.append(row)
    return rows


def method_contrasts(cells):
    rows=[]
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2,"pooled"):
            seeds=(0,1,2) if seed=="pooled" else (seed,)
            for method,reference in METHOD_PAIRS:
                row=dict(distribution=dist,master_seed=seed,method=method,reference=reference,n=20,K=1,B=4999,alpha=.05,original_total=30000 if seed=="pooled" else 10000)
                for metric in PAIR_METRICS:
                    difference=_values(cells,dist,seeds,method,metric)-_values(cells,dist,seeds,reference,metric)
                    for prefix,array in (("",difference),("conditional_",difference[np.isfinite(difference)])):
                        stats=mean_t_interval(array,confidence=.99)
                        for key,value in stats.items():row[f"{prefix}{metric}_{key}"]=value
                rows.append(row)
    return rows


def assemble_hypotheses(cells):
    quantities={};seed_details=[]
    for name,metric in (("H1","C"),("H2","T")):
        one,zero=(_values(cells,dist,(0,1,2),"bca",metric) for dist in ("lognormal","gamma"))
        stats=welch_interval(one,zero,confidence=.99)
        decision=practical_difference_status(stats["low"],stats["high"],margin=.01,complete=bool(stats["valid"]))
        quantities[name]={**stats,**decision,"metric":metric,"margin":.01,"group1":"lognormal","group0":"gamma","independent_groups":True,
                          "conditional_welch":welch_interval(one[np.isfinite(one)],zero[np.isfinite(zero)],confidence=.99)}
    for seed in (0,1,2):
        seed_details.append(dict(master_seed=seed,**{name:welch_interval(expected_values(cells,"lognormal",seed,"bca",metric),expected_values(cells,"gamma",seed,"bca",metric),confidence=.99) for name,metric in (("H1","C"),("H2","T"))}))
    return dict(question_id="RQ-M05",confirmatory_quantities=2,decision_confidence=.99,margin=.01,target_method="bca",B=4999,
                independent_originals_per_population=30000,seed_direction_gate=False,quantities=quantities,seed_details=seed_details,
                hypotheses={name:{key:quantities[name][key] for key in ("status","direction","equivalence_supported","reason")} for name in quantities},
                both_metrics_equivalent=all(q["equivalence_supported"] for q in quantities.values()))


@contextmanager
def metric_writer(path):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="",mode="wb",compresslevel=6,mtime=0,fileobj=raw) as zipped:
            with io.TextIOWrapper(zipped,encoding="utf-8",newline="") as text:
                writer=csv.DictWriter(text,fieldnames=CSV_FIELDS);writer.writeheader();yield writer


def write_metric_rows(writer,dist,seed,panel,metrics):
    for i,outer in enumerate(panel["outer_id"]):
        for mi,method in enumerate(METHODS):
            code=int(panel["reason_codes"][i,mi]);row=dict(distribution=dist,master_seed=seed,outer_id=int(outer),method=method,
                                                         complete=bool(metrics["complete"][i,mi]),reason_code=code,reason=str(panel["reason_labels"][code]))
            row.update({key:metrics[key][i,mi].item() for key in ("valid_inner","total_inner","invalid_inner",*METRICS)})
            writer.writerow(row)


def save_panel(path,panel,distribution,seed,R,B,namespace):
    with path.open("xb") as handle:
        np.savez_compressed(handle,**panel,methods=np.array(METHODS),distribution=distribution,master_seed=seed,n=20,R=R,observed_original_count=len(panel["outer_id"]),K=1,B=B,alpha=.05,namespace=namespace)


def write_figure(pooled,contrasts,output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    output.mkdir(exist_ok=False);fig=make_subplots(rows=1,cols=2,subplot_titles=["Coverage: 95% inference intervals, 99% MC bars","BCa independent differences L-G; +/-1pp band"])
    for dist in DISTRIBUTIONS:
        rows=[next(r for r in pooled if r["distribution"]==dist and r["method"]==m) for m in METHODS]
        fig.add_trace(go.Scatter(x=list(METHODS),y=[r["C"] for r in rows],mode="markers",name=dist,
                                error_y=dict(type="data",symmetric=False,array=[r["C_decision_high"]-r["C"] for r in rows],arrayminus=[r["C"]-r["C_decision_low"] for r in rows])),row=1,col=1)
    primary=next(r for r in contrasts if r["master_seed"]=="pooled" and r["method"]=="bca")
    fig.add_trace(go.Scatter(x=[primary[f"{m}_difference"] for m in ("C","T")],y=["coverage C","direction T"],mode="markers",name="BCa L-G",
                            error_x=dict(type="data",symmetric=False,array=[primary[f"{m}_high"]-primary[f"{m}_difference"] for m in ("C","T")],arrayminus=[primary[f"{m}_difference"]-primary[f"{m}_low"] for m in ("C","T")])),row=1,col=2)
    fig.add_vrect(x0=-.01,x1=.01,fillcolor="gray",opacity=.15,line_width=0,row=1,col=2);fig.add_vline(x=0,line_dash="dot",row=1,col=2)
    fig.update_layout(template="plotly_white",width=1250,height=600,title="RQ-M05: same mean/variance/skewness, independent populations; N30000 each, n20,K1,B4999")
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
    m=json.loads(manifest_path.read_text());validate_manifest(m);verify_committed_inputs(manifest_path,m,source_commit)
    output=output.resolve()
    if ROOT not in output.parents:raise ValueError("Output must be inside project")
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ("attempt.json","resources.json","panels","original_metrics.csv.gz","summary.csv","figures")):raise FileExistsError("Existing attempt/results preserved")
    attempt=dict(question_id="RQ-M05",status="started",started_utc=datetime.now(timezone.utc).isoformat(),source_git_sha=source_commit,manifest_sha256=sha256(manifest_path),namespace=MAIN_NAMESPACE,single_main_attempt=True)
    _write_json(output/"attempt.json",attempt);(output/"panels").mkdir()
    cells,completed={},[];resources={**attempt,"environment":runtime_environment(),"R":10000,"K":1,"B":4999,"n":20,"total_cells":6,"full_bootstrap_saved":False,"original_samples_saved":True,"max_resample_elements":1000000,
                                  "wall_scope":"module start before heavy imports through outputs, excluding interpreter startup/final resource write; outer guard and time measure full CLI"}
    def check():enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output))
    try:
        with metric_writer(output/"original_metrics.csv.gz") as writer:
            for dist,seed in fixed_cells():
                resources["active_cell"]=[dist,seed];progress={}
                try:panel=run_cell(dist,seed,check=check,progress=progress)
                except Exception:
                    observed=progress.get("observed_original_count",0);resources["partial_original_count"]=observed
                    if observed:
                        partial={key:array[:observed] for key,array in progress["panel"].items()}
                        partial["reason_labels"]=np.array(progress["reason_labels"],dtype="U96")
                        save_panel(output/"panels"/f"{dist}_seed{seed}_partial.npz",partial,dist,seed,10000,4999,MAIN_NAMESPACE)
                        write_metric_rows(writer,dist,seed,partial,panel_metrics(partial))
                    raise
                metrics=panel_metrics(panel);save_panel(output/"panels"/f"{dist}_seed{seed}.npz",panel,dist,seed,10000,4999,MAIN_NAMESPACE)
                write_metric_rows(writer,dist,seed,panel,metrics);cells[(dist,seed)]=metrics;completed.append([dist,seed]);check()
        seed_summary,pooled=summarize(cells),summarize(cells,pooled=True)
        populations,methods=population_contrasts(cells),method_contrasts(cells)
        write_csv(output/"summary.csv",seed_summary);write_csv(output/"pooled_summary.csv",pooled)
        write_csv(output/"population_contrasts.csv",populations);write_csv(output/"paired_method_contrasts.csv",methods)
        _write_json(output/"hypotheses.json",assemble_hypotheses(cells));write_figure(pooled,populations,output/"figures")
        check();resources["status"]="completed"
    except Exception as exc:
        resources.update(status="stopped",stop_reason=type(exc).__name__,stop_detail=str(exc) if isinstance(exc,BudgetExceeded) else "See preserved diagnostics; no automatic retry")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGTERM,signal.SIG_IGN)
        resources.update(completed_cells=len(completed),completed_cell_keys=completed,incomplete_cells=[list(c) for c in fixed_cells() if list(c) not in completed],finished_utc=datetime.now(timezone.utc).isoformat())
        _final_resources(output,resources)
    return resources


def run_smoke(output):
    output=output.resolve()
    if ROOT not in output.parents:raise ValueError("Smoke output must be inside project")
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError("Existing smoke attempt preserved")
    attempt=dict(question_id="RQ-M05",scope="smoke only; no scientific decisions",R=2,K=1,B=99,namespace=SMOKE_NAMESPACE,source_files_sha256={name:sha256(ROOT/name) for name in SOURCES})
    _write_json(output/"attempt.json",attempt);result,completed={**attempt,"environment":runtime_environment(),"status":"started"},[]
    try:
        for dist,seed in fixed_cells():
            panel=run_cell(dist,seed,R=2,B=99,namespace=SMOKE_NAMESPACE,check=lambda:enforce_budget(perf_counter()-_PROCESS_START,peak_rss_bytes(),directory_bytes(output)))
            completed.append(dict(distribution=dist,master_seed=seed,valid_entries=int(panel["valid"].sum())))
        result["status"]="completed"
    except Exception as exc:result.update(status="stopped",stop_reason=type(exc).__name__);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGTERM,signal.SIG_IGN)
        result.update(completed_cells=len(completed),cells=completed,original_samples=2*len(completed),inner_streams=2*len(completed),
                      bootstrap_samples=99*2*len(completed),logical_resampling_indices=20*99*2*len(completed));_final_resources(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-manifest",action="store_true");modes.add_argument("--run",action="store_true");modes.add_argument("--smoke",action="store_true")
    parser.add_argument("--manifest",type=Path,default=ROOT/"evidence/rq_m05/manifest.json");parser.add_argument("--output",type=Path);parser.add_argument("--source-commit")
    args=parser.parse_args()
    if args.prepare_manifest:
        path=args.manifest.resolve()
        if ROOT not in path.parents:parser.error("Manifest must be inside project")
        path.parent.mkdir(parents=True,exist_ok=True);_write_json(path,make_manifest());print("Prepared",path.relative_to(ROOT))
    elif args.smoke:
        result=run_smoke(args.output or ROOT/"evidence/rq_m05/smoke");print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))
    else:
        if not args.source_commit:parser.error("--run requires full reviewed --source-commit")
        result=run_benchmark(args.manifest.resolve(),args.output or ROOT/"evidence/rq_m05",args.source_commit);print(json.dumps({key:result[key] for key in ("status","completed_cells","process_wall_seconds","process_peak_rss_bytes")}))


if __name__=="__main__":main()
