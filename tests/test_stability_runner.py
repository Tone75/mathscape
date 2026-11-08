"""Deterministic RQ-M03 runner fixtures; never run main or smoke experiments."""

from copy import deepcopy
import hashlib
import json
import struct

import numpy as np
import pytest

from scripts import run_stability_benchmark as runner


def test_fixed_totals_namespaces_and_source_scope():
    cells = runner.fixed_cells()
    assert len(cells) == len(set(cells)) == 12
    assert len(cells)*100 == 1200
    assert len(cells)*100*20 == 24000
    assert len(cells)*100*20*4999 == 119976000
    assert sum(n*100*20*4999 for _,n,_ in cells) == 1499700000
    assert runner.MAIN_NAMESPACE == 0x4D3033
    assert runner.SMOKE_NAMESPACE == 0x534D3033
    assert runner.MAIN_NAMESPACE != runner.SMOKE_NAMESPACE
    assert "scripts/audit_stability.py" in runner.SOURCES
    assert "mathscape/bootstrap.py" in runner.SOURCES


def test_original_hash_is_fixed_little_endian_float64_bytes():
    values = np.array([1., -2., .25], dtype=">f8")
    expected = hashlib.sha256(struct.pack("<ddd", 1., -2., .25)).hexdigest()
    assert runner.sample_hash(values) == expected


def test_manifest_rejects_design_and_policy_changes():
    manifest = dict(schema_version=1,question_id="RQ-M03",design=deepcopy(runner.DESIGN),policy=deepcopy(runner.POLICY))
    runner.validate_manifest(manifest)
    manifest["design"]["R"] = 101
    with pytest.raises(ValueError):
        runner.validate_manifest(manifest)
    manifest["design"] = deepcopy(runner.DESIGN)
    manifest["policy"]["wall_seconds"] = 301
    with pytest.raises(ValueError):
        runner.validate_manifest(manifest)


def _sample(dist,n,**kwargs):
    return np.array([-1., -.2, 0., .4, 1.1]) + kwargs["outer_id"]*.01


def _means(sample,dist,n,**kwargs):
    return np.linspace(-.7,.8,kwargs["Bmax"]) + sample.mean() + kwargs["inner_id"]*.005


def test_tiny_deterministic_panel_uses_one_max_prefix_stream_per_inner():
    calls = []
    def means(sample,dist,n,**kwargs):
        calls.append(kwargs.copy())
        return _means(sample,dist,n,**kwargs)
    data,samples,hashes,rows = runner.run_cell("normal",5,1,R=2,K=4,Bs=(3,7,11),namespace=runner.SMOKE_NAMESPACE,
                                              sample_fn=_sample,bootstrap_fn=means)
    assert data["lower"].shape == (2,3,3,4)
    assert len(calls) == 8
    assert all(c["Bmax"] == 11 for c in calls)
    assert all(c["namespace"] == runner.SMOKE_NAMESPACE for c in calls)
    assert [c["inner_id"] for c in calls] == list(range(4))*2
    assert samples.shape == (2,5)
    assert hashes.tolist() == [runner.sample_hash(s) for s in samples]
    assert len(rows) == 18
    for r in rows:
        assert r["total_inner"] == (1 if r["method"] == "t" else 4)
        if r["method"] == "t":
            assert r["A"] == r["lower_sd"] == r["upper_sd"] == 0
    np.testing.assert_array_equal(data["lower"][:,:,0],np.repeat(data["lower"][:,:,0,:1],4,axis=2))


@pytest.mark.parametrize("values", [np.ones(5),np.array([np.nan,0,1,2,3])])
def test_invalid_original_is_retained_without_resampling(values):
    def sample(*args,**kwargs):
        return values
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid originals must not be resampled")
    data,samples,hashes,rows = runner.run_cell("normal",5,0,R=1,K=2,Bs=(3,7,11),sample_fn=sample,bootstrap_fn=forbidden)
    assert not data["valid"].any()
    assert (data["reason"] != "").all()
    assert all(not r["complete"] and np.isnan(r["A"]) and np.isnan(r["C"]) for r in rows)
    assert all(r["invalid_inner"] == r["total_inner"] for r in rows)


def test_defined_bootstrap_numerical_failure_does_not_replace_methods():
    def nonfinite(*args,**kwargs):
        raise ValueError("bootstrap_mean_nonfinite")
    data,_,_,rows = runner.run_cell("normal",5,0,R=1,K=2,Bs=(3,7,11),sample_fn=_sample,bootstrap_fn=nonfinite)
    assert data["valid"][:,:,0].all()
    assert not data["valid"][:,:,1:].any()
    assert (data["reason"][:,:,1:] == "bootstrap_mean_nonfinite").all()
    assert all(r["complete"] for r in rows if r["method"] == "t")


def test_unexpected_bootstrap_error_stops_instead_of_silent_fallback():
    def broken(*args,**kwargs):
        raise ValueError("configuration or implementation bug")
    with pytest.raises(ValueError,match="configuration"):
        runner.run_cell("normal",5,0,R=1,K=2,Bs=(3,7,11),sample_fn=_sample,bootstrap_fn=broken)


def test_budget_callback_stops_before_generating_inner_arrays():
    calls = []
    def stop():
        calls.append(1)
        if len(calls)==2:
            raise runner.BudgetExceeded("fixture limit")
    def forbidden(*args,**kwargs):
        raise AssertionError("budget must be checked before resampling")
    with pytest.raises(runner.BudgetExceeded):
        runner.run_cell("normal",5,0,R=1,K=2,Bs=(3,7,11),check=stop,sample_fn=_sample,bootstrap_fn=forbidden)


@pytest.mark.parametrize("wall,rss,size",[(300,1,1),(1,2*1024**3,1),(1,1,50*1024**2)])
def test_each_resource_limit_stops(wall,rss,size):
    with pytest.raises(runner.BudgetExceeded):
        runner.enforce_budget(wall,rss,size)


def _original_fixture():
    rows=[]
    for dist,n,seed in runner.fixed_cells():
        for outer in range(100):
            for B in (199,999,4999):
                for method in runner.METHODS:
                    A = {199:.6,999:.3,4999:.15}[B]+outer*.0001
                    C = .8+(outer%5)*.01
                    if method=="t":
                        A=0.
                        C=float(outer%20!=0)
                    W=outer%2
                    rows.append(dict(distribution=dist,n=n,master_seed=seed,outer_id=outer,B=B,method=method,
                                     R=100,K=20,alpha=.05,se=1.,sample_sha256="fixture",complete=True,
                                     valid_inner=1 if method=="t" else 20,total_inner=1 if method=="t" else 20,invalid_inner=0,
                                     lower_sd=A,upper_sd=A,A=A,C=C,mean_width=2.+outer*.001,
                                     miss_above=(1-C)/2,miss_below=(1-C)/2,warning_rate=.5 if W else .1,
                                     h2_class_complete=True,h2_eval_complete=True,h2_class_warning_count=5 if W else 0,
                                     h2_W=W,h2_A_eval=.1+.05*W+outer*.0001))
    return rows


def test_summary_counts_t_once_and_clusters_bootstrap_by_original():
    originals=_original_fixture()
    summary=runner.summarize_rows(originals)
    pooled=runner.summarize_rows(originals,pooled=True)
    assert len(summary)==108 and len(pooled)==36
    for r in summary:
        assert r["original_total"]==r["complete_original_count"]==100
        assert r["interval_count"]==(100 if r["method"]=="t" else 2000)
    r=next(r for r in pooled if r["distribution"]=="centered_exponential" and r["n"]==5 and r["B"]==4999 and r["method"]=="bca")
    C=np.array([.8+(outer%5)*.01 for seed in range(3) for outer in range(100)])
    assert r["original_total"]==300 and r["interval_count"]==6000
    assert r["coverage_mcse"]==pytest.approx(C.std(ddof=1)/np.sqrt(300))
    assert r["coverage_mcse"]!=pytest.approx(C.std(ddof=1)/np.sqrt(6000))


def test_four_confirmatory_quantities_and_seed_groups_from_original_fixture():
    hypotheses,groups=runner.assemble_hypotheses(_original_fixture())
    assert len(groups)==8
    assert set(hypotheses["quantities"])=={"H1_n5","H1_n20","H2","H3"}
    assert all(h["status"]=="supported" for h in hypotheses["hypotheses"].values())
    assert hypotheses["quantities"]["H2"]["N0"]==hypotheses["quantities"]["H2"]["N1"]==150


def test_h2_insufficient_seed_group_remains_inconclusive():
    originals=_original_fixture()
    for r in originals:
        r["h2_W"]=int(r["outer_id"]>=4)
    hypotheses,groups=runner.assemble_hypotheses(originals)
    assert hypotheses["quantities"]["H2"]["N0"]==12
    assert not hypotheses["quantities"]["H2"]["group_requirements_met"]
    assert hypotheses["hypotheses"]["H2"]["status"]=="inconclusive"


def test_h2_seed_group_shortage_records_reason_even_with_large_pooled_groups():
    originals=_original_fixture()
    for r in originals:
        if r["master_seed"]==0:
            r["h2_W"]=int(r["outer_id"]>=4)
    hypotheses,_=runner.assemble_hypotheses(originals)
    q=hypotheses["quantities"]["H2"]
    assert min(q["N0"],q["N1"])>=30
    assert q["conditional_welch"]["valid"]
    assert not q["valid"] and q["status"]=="inconclusive"
    assert q["reason"]=="insufficient_seed_group_count"


def test_partial_panel_cannot_support_using_smaller_denominator():
    originals=[r for r in _original_fixture() if not (r["distribution"]=="centered_exponential" and r["n"]==5 and r["master_seed"]==0 and r["outer_id"]==99)]
    hypotheses,_=runner.assemble_hypotheses(originals)
    for name in ("H1_n5","H3"):
        q=hypotheses["quantities"][name]
        assert q["N"]==300 and q["finite_count"]==299
        assert not q["complete_design"] and np.isnan(q["mean"]) and q["status"]=="inconclusive"
        assert q["conditional_interval"]["N"]==299
    summary=runner.summarize_rows(originals)
    target=next(r for r in summary if r["distribution"]=="centered_exponential" and r["n"]==5 and r["master_seed"]==0 and r["B"]==4999 and r["method"]=="bca")
    assert target["original_total"]==100 and target["observed_original_count"]==99 and target["missing_original_count"]==1
    assert np.isnan(target["coverage"]) and np.isfinite(target["conditional_coverage"])


def test_h1_clear_contrary_n_preserved_alongside_other_missing_n():
    originals=[r for r in _original_fixture() if not (r["distribution"]=="centered_exponential" and r["n"]==5 and r["outer_id"]==99)]
    for r in originals:
        if r["n"]==20 and r["B"]==4999:
            r["A"]+=1
    hypotheses,_=runner.assemble_hypotheses(originals)
    assert hypotheses["hypotheses"]["H1"]==dict(status="not_supported",n5="inconclusive",n20="not_supported")


def test_missing_h2_original_keeps_expected_denominator_and_na_primary_groups():
    originals=[r for r in _original_fixture() if not (r["distribution"]=="centered_exponential" and r["n"]==20 and r["master_seed"]==2 and r["outer_id"]==99 and r["B"]==999 and r["method"]=="bca")]
    hypotheses,groups=runner.assemble_hypotheses(originals)
    quantity=hypotheses["quantities"]["H2"]
    assert quantity["status"]=="inconclusive" and not quantity["valid"]
    assert np.isnan(quantity["difference"]) and quantity["reason"]
    for row in groups:
        if row["master_seed"]==2:
            assert row["original_total"]==100 and row["observed_original_count"]==99 and row["missing_original_count"]==1
            assert not row["all_required_complete"] and np.isnan(row["A_eval_mean"])
            assert np.isfinite(row["conditional_A_eval_mean"])


def test_existing_results_refused_before_any_attempt_or_generation(monkeypatch):
    class Manifest:
        def read_text(self):
            return json.dumps(dict(schema_version=1,question_id="RQ-M03",design=runner.DESIGN,policy=runner.POLICY))
    class Existing:
        parents=(runner.ROOT,)
        def resolve(self): return self
        def mkdir(self,**kwargs): pass
        def __truediv__(self,name): return self
        def exists(self): return True
    monkeypatch.setattr(runner,"verify_committed_inputs",lambda *args: None)
    monkeypatch.setattr(runner,"run_cell",lambda *args,**kwargs: pytest.fail("must not generate"))
    with pytest.raises(FileExistsError):
        runner.run_benchmark(Manifest(),Existing(),"a"*40)
