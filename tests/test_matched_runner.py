"""Deterministic runner fixtures only: never run main or stochastic smoke."""

from copy import deepcopy
import csv
import gzip
import json

import numpy as np
import pytest

from scripts import run_matched_benchmark as runner


def test_fixed_population_ids_totals_and_capacity_contract():
    assert runner.DISTRIBUTIONS==("gamma","lognormal")
    assert len(runner.fixed_cells())==6
    assert 6*10000==60000 and 60000*4999==299940000 and 60000*4999*20==5998800000
    assert runner.MAIN_NAMESPACE==0x4D3035 and runner.SMOKE_NAMESPACE==0x534D3035
    assert "requirements.txt" in runner.SOURCES and "evidence/rq_m05/SPEC.md" in runner.SOURCES
    assert "evidence/rq_m05/population_checks.json" in runner.SOURCES


def test_fixed_manifest_rejects_drift():
    m=dict(schema_version=1,question_id="RQ-M05",design=deepcopy(runner.DESIGN),policy=deepcopy(runner.POLICY))
    runner.validate_manifest(m);m["design"]["K"]=2
    with pytest.raises(ValueError):runner.validate_manifest(m)


def _sample(*args,**kwargs):return np.linspace(-1,1,20)+kwargs["outer_id"]*.01


def _means(sample,*args,**kwargs):return np.linspace(-.7,.8,kwargs["Bmax"])+sample.mean()


def test_compact_deterministic_panel_and_all_K1_stability_NA():
    calls=[]
    def means(sample,*args,**kwargs):calls.append(kwargs.copy());return _means(sample,*args,**kwargs)
    panel=runner.run_cell("gamma",0,R=2,B=19,sample_fn=_sample,bootstrap_fn=means)
    assert panel["samples"].shape==(2,20) and panel["lower"].shape==(2,3)
    assert panel["sample_sha256"].dtype.itemsize==64 and panel["sample_sha256"].dtype.kind=="S"
    assert panel["reason_codes"].dtype==np.uint8 and panel["outer_id"].dtype==np.uint16
    assert all(c["inner_id"]==0 and c["Bmax"]==19 for c in calls) and len(calls)==2
    metrics=runner.panel_metrics(panel)
    for key in runner.STABILITY:assert np.isnan(metrics[key]).all()
    assert metrics["total_inner"].tolist()==[[1,1,1],[1,1,1]]
    for i in range(2):assert panel["sample_sha256"][i].decode()==runner.sample_hash(panel["samples"][i])


@pytest.mark.parametrize("sample",[np.ones(20),np.full(20,np.nan)])
def test_invalid_original_is_reason_labelled_not_noncoverage(sample):
    def invalid(*args,**kwargs):return sample
    def forbidden(*args,**kwargs):raise AssertionError("must not generate bootstrap")
    panel=runner.run_cell("gamma",0,R=1,B=19,sample_fn=invalid,bootstrap_fn=forbidden)
    assert not panel["valid"].any() and (panel["reason_codes"]>0).all()
    metrics=runner.panel_metrics(panel)
    assert np.isnan(metrics["C"]).all() and np.isnan(metrics["T"]).all()


def test_numeric_failure_retains_t_without_fallback():
    def bad(*args,**kwargs):raise ValueError("bootstrap_mean_nonfinite")
    panel=runner.run_cell("gamma",0,R=1,B=19,sample_fn=_sample,bootstrap_fn=bad)
    assert panel["valid"][0,0] and not panel["valid"][0,1:].any()
    assert str(panel["reason_labels"][panel["reason_codes"][0,1]])=="bootstrap_mean_nonfinite"


def test_streaming_csv_gzip_roundtrip_has_all_records_and_NA(tmp_path):
    panel=runner.run_cell("gamma",0,R=2,B=19,sample_fn=_sample,bootstrap_fn=_means)
    metrics=runner.panel_metrics(panel);path=tmp_path/"records.csv.gz"
    with runner.metric_writer(path) as writer:runner.write_metric_rows(writer,"gamma",0,panel,metrics)
    with gzip.open(path,"rt",newline="") as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==6 and list(rows[0])==list(runner.CSV_FIELDS)
    assert all(row["S"]==row["Ase"]==row["Apop"]=="nan" for row in rows)
    with pytest.raises(FileExistsError):
        with runner.metric_writer(path):pass


def _cells():
    cells={}
    for dist,seed in runner.fixed_cells():
        lo=np.full((10000,3),-1.);hi=np.ones((10000,3));hi[np.arange(10000)%10==0,:]=-.1
        cells[(dist,seed)]=runner.panel_metrics(dict(lower=lo,upper=hi,valid=np.ones((10000,3),bool),tail_warning=np.zeros((10000,3),bool),outer_id=np.arange(10000,dtype=np.uint16)))
    return cells


def test_full_planned_denominators_and_original_MCSE():
    cells=_cells();summary=runner.summarize(cells);pooled=runner.summarize(cells,pooled=True)
    assert len(summary)==18 and len(pooled)==6
    row=pooled[0];values=np.tile((np.arange(10000)%10!=0).astype(float),3)
    assert row["original_total"]==row["valid_count"]==row["interval_count"]==30000
    assert row["C_mcse"]==pytest.approx(values.std(ddof=1)/np.sqrt(30000))
    assert not row["stability_applicable"]
    assert all(np.isnan(row[key]) for key in runner.STABILITY)


def test_equivalence_is_interval_containment_not_seed_gate():
    cells=_cells()
    # Put seed differences on opposite sides of zero while retaining pooled0.
    for seed,delta in ((0,.01),(1,-.005),(2,-.005)):
        cells[("lognormal",seed)]["C"]+=delta
        cells[("lognormal",seed)]["T"]-=delta
    h=runner.assemble_hypotheses(cells)
    assert not h["seed_direction_gate"]
    assert h["both_metrics_equivalent"]
    assert all(q["status"]=="not_supported" and q["equivalence_supported"] for q in h["quantities"].values())
    assert h["seed_details"][0]["H1"]["difference"]>0 and h["seed_details"][1]["H1"]["difference"]<0


def test_practical_difference_support_has_fixed_L_minus_G_direction():
    cells=_cells();cells[("lognormal",0)]["C"]+=.1;cells[("lognormal",1)]["C"]+=.1;cells[("lognormal",2)]["C"]+=.1
    h=runner.assemble_hypotheses(cells)
    assert h["quantities"]["H1"]["status"]=="supported" and h["quantities"]["H1"]["direction"]=="positive"


def test_other_method_invalid_does_not_poison_complete_BCa_target():
    cells=_cells();cells[("gamma",0)]["complete"][0,0]=False
    h=runner.assemble_hypotheses(cells)
    assert h["both_metrics_equivalent"]
    row=next(r for r in runner.summarize(cells) if r["distribution"]=="gamma" and r["master_seed"]==0 and r["method"]=="t")
    assert np.isnan(row["C"]) and row["invalid_count"]==1 and row["conditional_original_count"]==9999


def test_target_invalid_or_missing_keeps_N30000_and_holds():
    cells=_cells();cell=cells[("gamma",2)]
    for key,value in cell.items():cell[key]=value[:-1]
    h=runner.assemble_hypotheses(cells)
    for q in h["quantities"].values():
        assert q["N0"]==30000 and q["finite_count0"]==29999
        assert not q["valid"] and q["status"]=="inconclusive" and np.isnan(q["difference"])
        assert q["conditional_welch"]["N0"]==29999


def test_all_descriptive_population_and_paired_method_contrasts():
    cells=_cells();population=runner.population_contrasts(cells);paired=runner.method_contrasts(cells)
    assert len(population)==12 and len(paired)==24
    r=next(r for r in population if r["master_seed"]=="pooled" and r["method"]=="bca")
    assert r["C_N1"]==r["C_N0"]==30000 and r["C_difference"]==0
    p=next(r for r in paired if r["master_seed"]=="pooled")
    assert p["C_N"]==30000 and p["C_mean"]==0 and p["C_mcse"]==0 and np.isnan(p["C_low"])


def test_partial_progress_preserves_only_completed_originals():
    progress={};calls=[]
    def check():
        calls.append(1)
        if len(calls)==3:raise runner.BudgetExceeded("fixture")
    with pytest.raises(runner.BudgetExceeded):runner.run_cell("gamma",0,R=4,B=19,check=check,sample_fn=_sample,bootstrap_fn=_means,progress=progress)
    assert progress["observed_original_count"]==2
    for i in range(2):assert progress["panel"]["sample_sha256"][i].decode()==runner.sample_hash(progress["panel"]["samples"][i])


@pytest.mark.parametrize("wall,rss,size",[(294,1,1),(1,2*1024**3,1),(1,1,50*1024**2)])
def test_budget_guards(wall,rss,size):
    with pytest.raises(runner.BudgetExceeded):runner.enforce_budget(wall,rss,size)


def test_existing_attempt_refuses_all_generation(monkeypatch):
    class Manifest:
        def read_text(self):return json.dumps(dict(schema_version=1,question_id="RQ-M05",design=runner.DESIGN,policy=runner.POLICY))
    class Existing:
        parents=(runner.ROOT,)
        def resolve(self):return self
        def mkdir(self,**kwargs):pass
        def __truediv__(self,key):return self
        def exists(self):return True
    monkeypatch.setattr(runner,"verify_committed_inputs",lambda *args:None)
    monkeypatch.setattr(runner,"run_cell",lambda *args,**kwargs:pytest.fail("must not generate"))
    with pytest.raises(FileExistsError):runner.run_benchmark(Manifest(),Existing(),"a"*40)
