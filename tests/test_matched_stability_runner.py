"""Saved-array fixtures only: no mother samples or random smoke/main panels."""

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import run_matched_stability_benchmark as runner


def _saved(R=2):
    samples=np.array([np.linspace(-1,1,20)+i*.01 for i in range(R)])
    return dict(samples=samples,outer_id=np.arange(R,dtype=np.uint16),sample_sha256=np.array([runner.sample_hash(s) for s in samples],dtype="S64"))


def _bootstrap(sample,*args,**kwargs):
    indices=((np.arange(kwargs["Bmax"])[:,None]+np.arange(20)[None,:]%3+kwargs["inner_id"])%20).astype(np.uint8)
    means=np.mean(sample[indices],axis=1)
    return (means,indices) if kwargs["return_indices"] else means


def test_fixed_saved_selection_and_logical_processing():
    assert len(runner.fixed_cells())==6 and runner.DESIGN["outer_ids"]==list(range(200))
    assert 6*200==1200 and 1200*30==36000 and 36000*4999==179964000
    assert 36000*4999*20==3599280000 and 1200*30*2*3==216000
    assert runner.MAIN_NAMESPACE==0x4D3036 and runner.SMOKE_NAMESPACE==0x534D3036
    assert "mathscape/matched_moments.py" in runner.SOURCES


def test_manifest_rejects_selection_or_policy_changes():
    m=dict(schema_version=1,question_id="RQ-M06",design=deepcopy(runner.DESIGN),policy=deepcopy(runner.POLICY))
    runner.validate_manifest(m);m["design"]["outer_ids"][0]=999
    with pytest.raises(ValueError):runner.validate_manifest(m)


def test_saved_originals_kept_exact_with_one_stream_for_all_prefixes():
    saved=_saved();calls=[]
    def stream(sample,*args,**kwargs):calls.append(kwargs.copy());return _bootstrap(sample,*args,**kwargs)
    panel,rows=runner.run_cell("gamma",0,saved,R=2,K=2,Bs=(7,19),namespace=runner.SMOKE_NAMESPACE,bootstrap_fn=stream)
    assert panel["lower"].shape==(2,2,3,2) and len(rows)==12
    np.testing.assert_array_equal(panel["samples"],saved["samples"])
    np.testing.assert_array_equal(panel["sample_sha256"],saved["sample_sha256"])
    assert len(calls)==4 and all(c["Bmax"]==19 for c in calls)
    assert all(c["namespace"]==runner.SMOKE_NAMESPACE and not c["return_indices"] for c in calls)
    for row in rows:
        if row["method"]=="t":assert row["total_inner"]==1 and row["S"]==row["Ase"]==row["Apop"]==row["Vse"]==0


def test_predeclared_audit_streams_saved_without_redraw():
    streams=[];calls=[]
    def stream(sample,*args,**kwargs):calls.append(kwargs.copy());return _bootstrap(sample,*args,**kwargs)
    runner.run_cell("lognormal",2,_saved(1),R=1,K=30,Bs=(7,19),bootstrap_fn=stream,audit_sink=streams.append)
    assert len(calls)==30 and len(streams)==3
    assert [s["inner_id"] for s in streams]==[0,1,29]
    for s in streams:
        assert s["indices"].dtype==np.uint8 and s["indices"].shape==(19,20)
        assert s["native_indices_dtype"]=="int64" and s["stored_indices_dtype"]=="uint8"
        np.testing.assert_array_equal(s["means"],_saved(1)["samples"][0][s["indices"]].mean(axis=1))


def test_input_hash_mismatch_stops_before_resampling():
    saved=_saved(1);saved["sample_sha256"][0]=b"wrong"
    with pytest.raises(ValueError,match="hash mismatch"):
        runner.run_cell("gamma",0,saved,R=1,K=2,Bs=(7,19),bootstrap_fn=lambda *args,**kwargs:pytest.fail("must not resample"))


def test_invalid_saved_original_retains_reason_without_bootstrap():
    saved=_saved(1);saved["samples"][0]=1.;saved["sample_sha256"][0]=runner.sample_hash(saved["samples"][0])
    panel,rows=runner.run_cell("gamma",0,saved,R=1,K=2,Bs=(7,19),bootstrap_fn=lambda *args,**kwargs:pytest.fail("must not resample"))
    assert not panel["valid"].any() and (panel["reason_codes"]>0).all()
    assert all(not r["complete"] and np.isnan(r["Ase"]) and np.isnan(r["C"]) for r in rows)


def _originals():
    rows=[]
    for dist,seed in runner.fixed_cells():
        for outer in range(200):
            for B in (999,4999):
                for method in runner.METHODS:
                    A=(.2 if B==999 else .1)+(outer%7)*.001+(.001 if dist=="lognormal" else 0)
                    S=A/np.sqrt(20);C=.85+(outer%3)*.02
                    if method=="t":A=S=0.
                    rows.append(dict(distribution=dist,master_seed=seed,outer_id=outer,n=20,R=200,K=30,B=B,alpha=.05,method=method,se=1/np.sqrt(20),sample_sha256="fixture",
                                     complete=True,total_inner=1 if method=="t" else 30,valid_inner=1 if method=="t" else 30,invalid_inner=0,
                                     lower_sd=S,upper_sd=S,S=S,Ase=A,Apop=A,Vse=A*A,C=C,Mbelow=(1-C)*.8,Mabove=(1-C)*.2,T=(1-C)*.6,mean_width=1+outer*.001,warning_rate=.1+(outer%2)*.1))
    return rows


def test_summaries_count_original_units_and_preregistered_linear_quantiles():
    rows=_originals();summary=runner.summarize(rows);pooled=runner.summarize(rows,pooled=True)
    assert len(rows)==7200 and len(summary)==36 and len(pooled)==12
    for r in summary:assert r["interval_count"]==(200 if r["method"]=="t" else 6000)
    target=next(r for r in pooled if r["distribution"]=="gamma" and r["B"]==4999 and r["method"]=="bca")
    values=np.tile([.1+(i%7)*.001 for i in range(200)],3)
    assert target["original_total"]==600 and target["interval_count"]==18000
    assert target["Ase_mcse"]==pytest.approx(values.std(ddof=1)/np.sqrt(600))
    for q in runner.QUANTILES:assert target[f"Ase_p{int(q*100):02d}"]==pytest.approx(np.quantile(values,q,method="linear"))


def test_hypothesis_units_joint_reduction_and_no_seed_gate():
    h=runner.assemble_hypotheses(_originals())
    assert h["quantities"]["H1"]["N1"]==h["quantities"]["H1"]["N0"]==600
    assert h["hypotheses"]["H1"]["status"]=="not_supported" and h["hypotheses"]["H1"]["equivalence_supported"]
    assert "not percentage points" in h["quantities"]["H1"]["units"]
    assert h["hypotheses"]["H2"]==dict(status="supported",gamma="supported",lognormal="supported")
    assert not h["seed_direction_gate"]


def test_local_missing_gates_preserve_complete_other_targets():
    rows=[r for r in _originals() if not (r["distribution"]=="gamma" and r["master_seed"]==2 and r["outer_id"]==199 and r["B"]==999 and r["method"]=="bca")]
    h=runner.assemble_hypotheses(rows)
    assert h["hypotheses"]["H1"]["equivalence_supported"]
    assert h["quantities"]["H2_gamma"]["N"]==600 and h["quantities"]["H2_gamma"]["finite_count"]==599
    assert h["hypotheses"]["H2"]==dict(status="inconclusive",gamma="inconclusive",lognormal="supported")
    assert np.isnan(h["quantities"]["H2_gamma"]["mean"])


def test_clear_other_population_contradiction_has_priority_with_missing():
    rows=[r for r in _originals() if not (r["distribution"]=="gamma" and r["outer_id"]==199 and r["B"]==999 and r["method"]=="bca")]
    for r in rows:
        if r["distribution"]=="lognormal" and r["B"]==4999:r["Ase"]+=.2
    h=runner.assemble_hypotheses(rows)
    assert h["hypotheses"]["H2"]==dict(status="not_supported",gamma="inconclusive",lognormal="not_supported")


def test_all_descriptive_contrasts_and_ratio_of_means():
    rows=_originals();pop=runner.population_contrasts(rows);B=runner.paired_contrasts(rows,"B");methods=runner.paired_contrasts(rows,"method")
    assert len(pop)==24 and len(B)==24 and len(methods)==48
    b=next(r for r in B if r["distribution"]=="gamma" and r["master_seed"]=="pooled" and r["method"]=="bca")
    low=np.array([.2+(i%7)*.001 for i in range(200)]);high=np.array([.1+(i%7)*.001 for i in range(200)])
    assert b["ratio_mean_Ase_4999_over_999"]==pytest.approx(high.mean()/low.mean())
    assert not np.isclose(b["ratio_mean_Ase_4999_over_999"],np.mean(high/low),rtol=1e-7,atol=1e-8)
    t=next(r for r in B if r["method"]=="t")
    assert np.isnan(t["ratio_mean_Ase_4999_over_999"])


def test_h2_zero_lower_boundary_is_not_supported():
    assert runner.reduction_status(0.,.1)["status"]=="not_supported"
    assert runner.reduction_status(-.1,0.)["status"]=="inconclusive"
    assert runner.combine_reduction_status(["not_supported","inconclusive"])=="not_supported"


def test_partial_progress_does_not_publish_uncomputed_originals():
    saved=_saved(3);progress={};calls=[]
    def check():
        calls.append(1)
        if len(calls)==5:raise runner.BudgetExceeded("fixture")
    with pytest.raises(runner.BudgetExceeded):runner.run_cell("gamma",0,saved,R=3,K=2,Bs=(7,19),bootstrap_fn=_bootstrap,check=check,progress=progress)
    assert progress["observed_original_count"]==1 and len(progress["rows"])==6


@pytest.mark.parametrize("wall,rss,size",[(294,1,1),(1,2*1024**3,1),(1,1,50*1024**2)])
def test_resource_guards(wall,rss,size):
    with pytest.raises(runner.BudgetExceeded):runner.enforce_budget(wall,rss,size)


def test_existing_attempt_refused_before_resampling(monkeypatch):
    class Manifest:
        def read_text(self):return json.dumps(dict(schema_version=1,question_id="RQ-M06",design=runner.DESIGN,policy=runner.POLICY,inputs={}))
    class Existing:
        parents=(runner.ROOT,)
        def resolve(self):return self
        def mkdir(self,**kwargs):pass
        def __truediv__(self,key):return self
        def exists(self):return True
    monkeypatch.setattr(runner,"verify_committed_inputs",lambda *args:None);monkeypatch.setattr(runner,"load_inputs",lambda *args,**kwargs:{})
    monkeypatch.setattr(runner,"run_cell",lambda *args,**kwargs:pytest.fail("must not resample"))
    with pytest.raises(FileExistsError):runner.run_benchmark(Manifest(),Existing(),"a"*40)
