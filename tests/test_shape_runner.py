"""Deterministic RQ-M04 runner tests; no stochastic smoke/main experiments."""

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import run_shape_benchmark as runner


def test_design_ids_and_processing_totals():
    assert runner.DISTRIBUTIONS==("normal","gamma_0.5","gamma_1","gamma_2","gamma_4","gamma_16")
    cells=runner.fixed_cells()
    assert len(cells)==len(set(cells))==18
    assert 18*400==7200 and 18*400*10==72000
    assert 18*400*10*4999==359928000
    assert 18*400*10*4999*20==7198560000
    assert runner.MAIN_NAMESPACE==0x4D3034 and runner.SMOKE_NAMESPACE==0x534D3034


def test_manifest_drift_rejected():
    m=dict(schema_version=1,question_id="RQ-M04",design=deepcopy(runner.DESIGN),policy=deepcopy(runner.POLICY))
    runner.validate_manifest(m)
    m["design"]["R"]+=1
    with pytest.raises(ValueError):runner.validate_manifest(m)


def _sample(*args,**kwargs):
    return np.linspace(-1,1,20)+kwargs["outer_id"]*.01


def _means(sample,*args,**kwargs):
    return sample.mean()+np.linspace(-.8,.8,kwargs["Bmax"])+kwargs["inner_id"]*.003


def test_tiny_panel_uses_one_maximum_prefix_and_original_hash():
    calls=[]
    def means(sample,*args,**kwargs):
        calls.append(kwargs.copy());return _means(sample,*args,**kwargs)
    data,samples,hashes,rows=runner.run_cell("normal",0,R=2,K=2,Bs=(7,19),sample_fn=_sample,bootstrap_fn=means)
    assert data["lower"].shape==(2,2,3,2) and samples.shape==(2,20)
    assert len(calls)==4 and all(c["Bmax"]==19 for c in calls)
    assert [c["inner_id"] for c in calls]==[0,1,0,1]
    assert hashes.tolist()==[runner.sample_hash(sample) for sample in samples]
    assert len(rows)==12
    for row in rows:
        if row["method"]=="t":
            assert row["total_inner"]==1
            assert row["S"]==row["Ase"]==row["Apop"]==0


@pytest.mark.parametrize("sample",[np.ones(20),np.full(20,np.nan)])
def test_invalid_original_preserved_without_bootstrap(sample):
    def invalid(*args,**kwargs):return sample
    def forbidden(*args,**kwargs):raise AssertionError("must not resample invalid original")
    data,_,_,rows=runner.run_cell("gamma_0.5",0,R=1,K=2,Bs=(7,19),sample_fn=invalid,bootstrap_fn=forbidden)
    assert not data["valid"].any() and (data["reason"]!="").all()
    assert all(not r["complete"] and np.isnan(r["C"]) and np.isnan(r["T"]) for r in rows)


def test_known_numeric_failure_retains_t_and_reason():
    def bad(*args,**kwargs):raise ValueError("bootstrap_mean_nonfinite")
    data,_,_,rows=runner.run_cell("normal",0,R=1,K=2,Bs=(7,19),sample_fn=_sample,bootstrap_fn=bad)
    assert data["valid"][:,:,0].all() and not data["valid"][:,:,1:].any()
    assert (data["reason"][:,:,1:]=="bootstrap_mean_nonfinite").all()


def test_unexpected_errors_stop_without_fallback():
    def bad(*args,**kwargs):raise ValueError("bug/configuration")
    with pytest.raises(ValueError,match="bug"):
        runner.run_cell("normal",0,R=1,K=2,Bs=(7,19),sample_fn=_sample,bootstrap_fn=bad)


def _original_fixture():
    rows=[]
    for distribution,seed in runner.fixed_cells():
        for outer in range(400):
            for B in (999,4999):
                for method in runner.METHODS:
                    C=(.86 if distribution=="gamma_0.5" else .90 if distribution=="gamma_16" else .94)+(outer%5-2)*.002
                    T=(.06 if distribution=="gamma_0.5" else .01)+(outer%5-2)*.001
                    S=(.02 if B==4999 else .05)+outer*.00001
                    if method=="t":S=0.
                    rows.append(dict(distribution=distribution,n=20,master_seed=seed,outer_id=outer,B=B,method=method,R=400,K=10,alpha=.05,se=.2,sample_sha256="fixture",
                                     complete=True,total_inner=1 if method=="t" else 10,valid_inner=1 if method=="t" else 10,invalid_inner=0,
                                     lower_sd=S,upper_sd=S,S=S,Ase=S/.2,Apop=S*np.sqrt(20),C=C,Mbelow=(1-C+T)/2,Mabove=(1-C-T)/2,T=T,mean_width=1.+outer*.001,warning_rate=.1))
    return rows


def test_summary_original_units_t_single_count_and_pooled_size():
    originals=_original_fixture()
    summary=runner.summarize_rows(originals);pooled=runner.summarize_rows(originals,pooled=True)
    assert len(summary)==108 and len(pooled)==36
    for r in summary:
        assert r["original_total"]==r["complete_original_count"]==400
        assert r["interval_count"]==(400 if r["method"]=="t" else 4000)
    r=next(r for r in pooled if r["distribution"]=="gamma_0.5" and r["B"]==4999 and r["method"]=="bca")
    c=np.array([.86+(outer%5-2)*.002 for seed in range(3) for outer in range(400)])
    assert r["original_total"]==1200 and r["interval_count"]==12000
    assert r["C_mcse"]==pytest.approx(c.std(ddof=1)/np.sqrt(1200))
    assert r["C_mcse"]!=pytest.approx(c.std(ddof=1)/np.sqrt(12000))


def test_two_confirmatory_quantities_are_independent_groups_and_original_t():
    h=runner.assemble_hypotheses(_original_fixture())
    assert h["hypotheses"]==dict(H1=dict(status="supported"),H2=dict(status="supported"))
    assert h["quantities"]["H1"]["N1"]==h["quantities"]["H1"]["N0"]==1200
    assert h["quantities"]["H2"]["N"]==1200
    assert h["quantities"]["H1"]["independent_groups"]


def test_zero_seed_threshold_differs_from_practical_pooled_threshold():
    originals=_original_fixture()
    for r in originals:
        if r["distribution"]=="gamma_0.5":
            r["C"]+=(.03 if r["master_seed"]==0 else -.03)
            r["T"]+=( -.05 if r["master_seed"]==0 else .02)
    h=runner.assemble_hypotheses(originals)
    assert h["quantities"]["H1"]["seed_points"][0]>-.02
    assert h["quantities"]["H1"]["seed_points"][0]<0
    assert h["quantities"]["H2"]["seed_points"][0]<.03
    assert h["quantities"]["H2"]["seed_points"][0]>0
    assert h["hypotheses"]["H1"]["status"]==h["hypotheses"]["H2"]["status"]=="supported"


def test_unrelated_population_or_B999_invalid_does_not_poison_confirmation():
    originals=_original_fixture()
    for r in originals:
        if r["distribution"]=="normal" or r["B"]==999:
            r.update(complete=False,C=np.nan,T=np.nan)
    h=runner.assemble_hypotheses(originals)
    assert all(q["status"]=="supported" for q in h["hypotheses"].values())


def test_missing_or_invalid_target_keeps_expected_denominator_and_holds_locally():
    originals=[r for r in _original_fixture() if not (r["distribution"]=="gamma_16" and r["master_seed"]==2 and r["outer_id"]==399 and r["B"]==4999 and r["method"]=="bca")]
    h=runner.assemble_hypotheses(originals)
    assert h["quantities"]["H1"]["N0"]==1200 and h["quantities"]["H1"]["finite_count0"]==1199
    assert np.isnan(h["quantities"]["H1"]["difference"])
    assert h["hypotheses"]["H1"]["status"]=="inconclusive"
    assert h["hypotheses"]["H2"]["status"]=="supported"
    summary=runner.summarize_rows(originals)
    target=next(r for r in summary if r["distribution"]=="gamma_16" and r["master_seed"]==2 and r["B"]==4999 and r["method"]=="bca")
    assert target["original_total"]==400 and target["missing_original_count"]==1
    assert np.isnan(target["C"]) and np.isfinite(target["conditional_C"])


def test_paired_descriptive_outputs_use_original_differences():
    originals=_original_fixture()
    B_rows=runner.descriptive_pairs(originals,"B");method_rows=runner.descriptive_pairs(originals,"method")
    assert len(B_rows)==72 and len(method_rows)==144
    target=next(r for r in B_rows if r["distribution"]=="gamma_0.5" and r["master_seed"]=="pooled" and r["method"]=="bca")
    assert target["Ase_difference"]==pytest.approx((.02-.05)/.2)
    assert target["C_difference"]==0
    assert np.isnan(target["C_decision_low"])
    assert target["C_complete_original_count"]==1200


@pytest.mark.parametrize("wall,rss,size",[(294,1,1),(1,2*1024**3,1),(1,1,50*1024**2)])
def test_resource_guards(wall,rss,size):
    with pytest.raises(runner.BudgetExceeded):runner.enforce_budget(wall,rss,size)


def test_budget_checked_before_inner_generation():
    checks=[]
    def check():
        checks.append(1)
        if len(checks)==2:raise runner.BudgetExceeded("fixture")
    def forbidden(*args,**kwargs):raise AssertionError("must stop before resampling")
    with pytest.raises(runner.BudgetExceeded):
        runner.run_cell("normal",0,R=1,K=2,Bs=(7,19),check=check,sample_fn=_sample,bootstrap_fn=forbidden)


def test_existing_attempt_refused_before_any_generation(monkeypatch):
    class Manifest:
        def read_text(self):return json.dumps(dict(schema_version=1,question_id="RQ-M04",design=runner.DESIGN,policy=runner.POLICY))
    class Existing:
        parents=(runner.ROOT,)
        def resolve(self):return self
        def mkdir(self,**kwargs):pass
        def __truediv__(self,name):return self
        def exists(self):return True
    monkeypatch.setattr(runner,"verify_committed_inputs",lambda *args:None)
    monkeypatch.setattr(runner,"run_cell",lambda *args,**kwargs:pytest.fail("must not generate"))
    with pytest.raises(FileExistsError):runner.run_benchmark(Manifest(),Existing(),"a"*40)
