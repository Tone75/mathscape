"""Deterministic budget/pairing/lineage references; no smoke/main draws."""

from copy import deepcopy
from io import BytesIO
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from mathscape import allocation as a
from mathscape import finite_k, matched_moments, stability


SAMPLE=np.array([0.,.1,.2,.2,.3,.5,.7,.9,1.,1.2,1.5,1.8,2.,2.5,3.,3.2,4.,5.,7.,9.])


def endpoints(scales=None,group_size=3):
    scales=np.asarray(scales if scales is not None else [[1,2,3,4],[2,3,4,5],[1,3,2,5]],float)
    r=len(scales);lo=np.empty((r,3,4*group_size));hi=np.empty_like(lo)
    pattern=np.linspace(-1,1,group_size)
    pattern=pattern/np.std(pattern,ddof=1)
    for repeat in range(r):
        for method in (1,2):
            lo[repeat,method]=np.concatenate([pattern*s for s in scales[repeat]])
            hi[repeat,method]=lo[repeat,method]+20
    lo[:,0],hi[:,0]=.8,1.8
    return dict(lower=lo,upper=hi,valid=np.ones(lo.shape,bool),tail_warning=np.zeros(lo.shape,bool))


def test_exact_M08_key_int64_indices_order_batch_and_no_original_generation():
    assert a.MAIN_NAMESPACE==0x4D3038 and a.SMOKE_NAMESPACE==0x534D3038
    assert a.DISTRIBUTION_IDS==dict(gamma=1,lognormal=2)
    assert not hasattr(a,"generate_sample")
    for dist,identifier in a.DISTRIBUTION_IDS.items():
        actual=a.rng_for_panel(dist,20,2,9,1,19,119)
        ref=np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=2,
            spawn_key=(a.MAIN_NAMESPACE,identifier,20,9,1,19,119))))
        np.testing.assert_array_equal(actual.bit_generator.random_raw(5),ref.bit_generator.random_raw(5))
    kw=dict(distribution="lognormal",master_seed=1,outer_id=3,budget_repeat_id=4,stream_id=59,
            Bmax=17,namespace=a.SMOKE_NAMESPACE)
    means,indices=a.generate_bootstrap_means(SAMPLE,**kw,return_indices=True)
    ref=np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=1,
        spawn_key=(a.SMOKE_NAMESPACE,2,20,3,1,4,59))))
    expected=ref.integers(0,20,size=(17,20),dtype=np.int64)
    assert indices.dtype==np.uint8
    np.testing.assert_array_equal(indices,expected)
    np.testing.assert_array_equal(means,SAMPLE[expected].mean(axis=1))
    a.rng_for_panel("gamma",budget_repeat_id=11,stream_id=73)
    for cap in (20,60,1_000_000):
        np.testing.assert_array_equal(means,a.generate_bootstrap_means(SAMPLE,**kw,max_elements=cap))
    assert not np.array_equal(means,a.generate_bootstrap_means(SAMPLE,**(kw|dict(budget_repeat_id=5))))


@pytest.mark.parametrize("kwargs",[dict(distribution="normal"),dict(distribution=[]),dict(n=19),
    dict(master_seed=3),dict(master_seed=True),dict(outer_id=10),dict(purpose_id=0),
    dict(budget_repeat_id=20),dict(budget_repeat_id=-1),dict(stream_id=120),
    dict(namespace=finite_k.MAIN_NAMESPACE)])
def test_invalid_rng_key_rejected(kwargs):
    with pytest.raises(ValueError):a.rng_for_panel(**({"distribution":"gamma"}|kwargs))


def test_native_dtype_memory_cap_and_no_t_recomputation(monkeypatch):
    calls=[]
    class Fixed:
        def integers(self,low,high,size,dtype):
            assert (low,high,dtype)==(0,20,np.int64)
            calls.append(size)
            return np.arange(np.prod(size),dtype=dtype).reshape(size)%20
    monkeypatch.setattr(a,"rng_for_panel",lambda *args,**kw:Fixed())
    monkeypatch.setattr(a,"prepare_sample",lambda *args,**kw:pytest.fail("recomputed t/a"))
    means,indices=a.generate_bootstrap_means(SAMPLE,"gamma",Bmax=7,max_elements=60,return_indices=True)
    assert calls==[(3,20),(3,20),(1,20)]
    np.testing.assert_array_equal(means,SAMPLE[indices].mean(axis=1))


@pytest.mark.parametrize("sample,kwargs",[(np.zeros(20),{}),(np.ones(19),{}),
    (np.full(20,np.nan),{}),(np.ones((2,10)),{}),(np.arange(20,dtype=complex),{}),
    (SAMPLE,dict(Bmax=5000)),(SAMPLE,dict(max_elements=19)),
    (SAMPLE,dict(max_elements=1000001)),(SAMPLE,dict(return_indices=1))])
def test_bad_resample_inputs_rejected(sample,kwargs):
    with pytest.raises(ValueError):a.generate_bootstrap_means(sample,"gamma",**kwargs)


def test_accepted_inference_contract_shared_SciPy_reference_and_truth_separation():
    assert a.prepare_sample is stability.prepare_sample and a.infer_single is finite_k.infer_single
    assert set(inspect.signature(a.infer_single).parameters)=={"context","means"}
    reference=stats.bootstrap((SAMPLE,),np.mean,n_resamples=4999,method="percentile",
        confidence_level=.95,rng=np.random.default_rng(74))
    bca=stats.bootstrap((SAMPLE,),np.mean,n_resamples=0,method="BCa",
        confidence_level=.95,bootstrap_result=reference)
    result=a.infer_single(a.prepare_sample(SAMPLE),reference.bootstrap_distribution)
    for name,expected in (("percentile",reference),("bca",bca)):
        assert [result[name]["lower"],result[name]["upper"]]==pytest.approx(
            [expected.confidence_interval.low,expected.confidence_interval.high],abs=1e-12)


def test_hand_groups_square_root_before_average_and_both_SS_partitions():
    raw={key:value[0] for key,value in endpoints([[1,2,3,4]]).items()}
    out=a.budget_metrics(raw,1.,group_size=3)
    groups,budget,decomp=out["groups"],out["allocations"],out["decomposition"]
    assert a.ALLOCATIONS==("P30","P60","P120") and a.ALLOCATION_J==(4,2,1)
    assert a.group_slices(3)==((0,3),(3,6),(6,9),(9,12),(0,6),(6,12),(0,12))
    np.testing.assert_allclose(groups["V"][2,:4],[1,4,9,16])
    np.testing.assert_allclose(groups["A"][2,:4],[1,2,3,4])
    assert budget["T"][0,2]==2.5
    assert budget["Vbar"][0,2]==7.5
    assert budget["T"][0,2]!=pytest.approx(np.sqrt(budget["Vbar"][0,2]))
    assert budget["T"][1,2]==pytest.approx((np.sqrt(2)+np.sqrt(10))/2)
    assert budget["T"][2,2]==pytest.approx(np.sqrt(60/11))
    assert decomp["Jgap"][2]==pytest.approx(np.sqrt(7.5)-2.5)
    assert decomp["Pgap"][2]==pytest.approx(np.sqrt(60/11)-np.sqrt(7.5))
    assert decomp["delta"][2]==pytest.approx(np.sqrt(60/11)-2.5)
    assert not any("Rgap" in key for key in decomp)
    assert decomp["algebra_ok"].all()
    for partition in (30,60):
        for end in ("lower","upper"):
            assert abs(decomp[f"ss{partition}_{end}_residual"][2])<a.IDENTITY_ATOL
    np.testing.assert_allclose(budget["Ase"],budget["T"])
    np.testing.assert_allclose(budget["Vse"],budget["Vbar"])


def test_P30_group_order_invariant_but_P60_has_fixed_adjacency():
    data=endpoints([[1,2,3,4]])
    raw={key:value[0] for key,value in data.items()}
    first=a.budget_metrics(raw,1.,group_size=3)
    order=np.array([0,2,1,3]);indices=np.concatenate([np.arange(3*g,3*g+3) for g in order])
    changed={key:value[:,indices] for key,value in raw.items()}
    second=a.budget_metrics(changed,1.,group_size=3)
    assert first["allocations"]["T"][0,2]==second["allocations"]["T"][0,2]
    assert first["allocations"]["T"][2,2]==pytest.approx(second["allocations"]["T"][2,2])
    assert first["allocations"]["T"][1,2]!=pytest.approx(second["allocations"]["T"][1,2])
    assert second["groups"]["A"][2,1]==3


def test_P30_measured_variance_not_first30_Q_div4_or_mean_group_variances_div4():
    data=endpoints([[1,4,2,3],[4,1,3,2],[1,3,4,2],[4,2,1,3]])
    out=a.original_metrics(data,1.,group_size=3,expected_repeats=4)
    first=out["budget"]["groups"]["A"][:,2,0]
    each=out["budget"]["groups"]["A"][:,2,:4]
    Q=out["original_allocations"]["Q"][0,2]
    assert Q==0
    assert np.var(first,ddof=1)/4>0
    assert np.var(each,axis=0,ddof=1).mean()/4>0
    np.testing.assert_array_equal(out["budget"]["allocations"]["T"][:,0,2],np.full(4,2.5))


def test_original_M_Q_d_L_U_are_paired_with_budget_repeat_R_not_original_count():
    out=a.original_metrics(endpoints(),.5,group_size=3,expected_repeats=3)
    T=out["budget"]["allocations"]["T"]
    original,comp=out["original_allocations"],out["original_comparisons"]
    assert a.ORIGINALS_PER_CELL==10 and a.BUDGET_REPEATS==20
    for method in (1,2):
        for p in range(3):
            assert original["M"][p,method]==pytest.approx(T[:,p,method].mean())
            assert original["Q"][p,method]==pytest.approx(np.var(T[:,p,method],ddof=1))
            assert original["Q_over_R"][p,method]==original["Q"][p,method]/3
        q30,q120=original["Q"][[0,2],method]
        assert comp["d"][method]==pytest.approx(np.mean(T[:,2,method]-T[:,0,method]))
        assert comp["L"][method]==pytest.approx(q120-.8*q30)
        assert comp["U"][method]==pytest.approx(q120-1.25*q30)
        assert comp["ratio_Q120_Q30"][method]==pytest.approx(q120/q30)
    assert comp["total_streams"].tolist()==[1,36,36]
    assert comp["planned_repeats"].tolist()==[3,3,3]


def test_t_exact_zero_all_allocations_variances_gaps_and_zero_ratio_NA():
    out=a.original_metrics(endpoints(),.5,group_size=3,expected_repeats=3)
    for key in ("V","A"):assert np.all(out["budget"]["groups"][key][:,0]==0)
    for key in ("T","Vbar","S","Ase","Vse"):assert np.all(out["budget"]["allocations"][key][:,:,0]==0)
    for key in a.DECOMPOSITION_FIELDS:assert np.all(out["budget"]["decomposition"][key][:,0]==0)
    for key in ("M","Q","Q_over_R"):assert np.all(out["original_allocations"][key][:,0]==0)
    assert np.isnan(out["original_comparisons"]["ratio_Q120_Q30"][0])
    assert not a.mean_t_interval(np.zeros(30))["valid"]
    old=matched_moments.original_metrics(dict(lower=np.full(3,-1.),upper=np.full(3,1.),
        valid=np.ones(3,bool),tail_warning=np.zeros(3,bool)))
    assert all(np.isnan(old[key]).all() for key in ("lower_sd","upper_sd","S","Ase","Apop"))


def test_invalid_local_method_all_R_streams_required_and_missing_t_does_not_poison_BCa():
    data=endpoints()
    data["valid"][2,1,11]=False
    data["valid"][1,0,10]=False;data["lower"][1,0,10]=np.nan
    data["tail_warning"][:,2,0]=True
    out=a.original_metrics(data,.5,group_size=3,expected_repeats=3)
    assert out["original_comparisons"]["complete"].tolist()==[False,False,True]
    assert out["original_comparisons"]["invalid_streams"].tolist()==[1,1,0]
    assert np.isnan(out["original_allocations"]["M"][:,:2]).all()
    assert np.isfinite(out["original_allocations"]["M"][:,2]).all()
    assert out["budget"]["allocations"]["warning_rate"][0,0,2]==1/12
    assert np.isnan(out["budget"]["groups"]["A"][2,1]).all()
    assert not out["budget"]["groups"]["complete"][2,1].any()
    np.testing.assert_allclose(out["budget"]["allocations"]["C"][:,:,2]+out["budget"]["allocations"]["Mbelow"][:,:,2]+out["budget"]["allocations"]["Mabove"][:,:,2],1)
    with pytest.raises(ValueError,match="planned budget"):
        a.original_metrics({k:v[:2] for k,v in data.items()},.5,group_size=3,expected_repeats=3)


def test_missing_budget_padding_strict_NA_and_group_counts_retained():
    data=endpoints();data["valid"][2]=False;data["lower"][2]=np.nan;data["upper"][2]=np.nan
    out=a.original_metrics(data,.5,group_size=3,expected_repeats=3)
    assert not out["original_comparisons"]["complete"].any()
    assert out["original_comparisons"]["complete_repeats"].tolist()==[2,2,2]
    assert out["original_comparisons"]["planned_repeats"].tolist()==[3,3,3]
    assert np.isnan(out["original_comparisons"]["d"]).all()
    assert out["original_comparisons"]["invalid_streams"].tolist()==[1,12,12]


@pytest.mark.parametrize("se",[0,-1,np.nan,np.inf,True,"0.5",10**1000])
def test_invalid_scale_never_silent_zero(se):
    out=a.original_metrics(endpoints(),se,group_size=3,expected_repeats=3)
    assert not out["original_comparisons"]["complete"].any()
    assert np.isnan(out["original_comparisons"]["L"]).all()


def test_bootstrap_scale_overflow_is_local_and_exact_t_zero_remains_defined():
    out=a.original_metrics(endpoints(),1e-300,group_size=3,expected_repeats=3)
    assert out["original_comparisons"]["complete"].tolist()==[True,False,False]
    assert out["original_comparisons"]["d"][0]==0
    assert np.isnan(out["original_comparisons"]["L"][1:]).all()


def test_main_shape_fixed_fixture_locks_inherited_tolerances_and_no_rng():
    data=endpoints([[1+.01*r,2+.03*r,3+.02*r,4+.01*r] for r in range(20)],group_size=30)
    out=a.original_metrics(data,.5)
    assert out["budget"]["groups"]["A"].shape==(20,3,7)
    assert out["budget"]["allocations"]["T"].shape==(20,3,3)
    assert out["budget"]["decomposition"]["algebra_ok"].all()
    assert out["original_comparisons"]["complete"].all()
    assert (a.NUMERIC_RTOL,a.NUMERIC_ATOL,a.IDENTITY_RTOL,a.IDENTITY_ATOL)==(2e-10,2e-12,2e-12,2e-12)
    for key in a.DECOMPOSITION_FIELDS:
        if key.endswith("normalized_residual"):
            assert np.max(np.abs(out["budget"]["decomposition"][key]))<a.IDENTITY_ATOL


def test_default995_original_CI_df29_and_missing_not_silently_omitted():
    values=np.linspace(-.001,.001,30)
    actual=a.mean_t_interval(values)
    expected=stats.ttest_1samp(values,0).confidence_interval(.995)
    assert actual["N"]==30 and actual["df"]==29
    assert [actual["low"],actual["high"]]==pytest.approx([expected.low,expected.high])
    values[0]=np.nan;bad=a.mean_t_interval(values)
    assert bad["N"]==30 and bad["finite_count"]==29 and not bad["valid"]


@pytest.mark.parametrize("bounds,status",[((-.002,.002),"supported"),
    ((.0026,.004),"not_supported"),((-.004,-.0026),"not_supported"),
    ((-.0025,.002),"inconclusive"),((-.002,.0025),"inconclusive"),
    ((0,0),"inconclusive")])
def test_H1_strict_band_and_SE0(bounds,status):
    assert a.band_status(*bounds)["status"]==status


@pytest.mark.parametrize("bounds,status",[((.01,.02,-.02,-.01),"supported"),
    ((-.02,0,-.02,-.01),"not_supported"),((.01,.02,0,.02),"not_supported"),
    ((0,.02,-.02,-.01),"inconclusive"),((.01,.02,-.02,0),"inconclusive"),
    ((-.02,-.01,0,0),"inconclusive"),((0,0,.01,.02),"inconclusive"),
    ((np.nan,.01,.01,.02),"inconclusive")])
def test_H2_all_four_zero_boundaries_and_necessary_interval_validity(bounds,status):
    assert a.variance_band_status(*bounds,mean_Q30=1)["status"]==status
    assert a.variance_band_status(*bounds,mean_Q30=1,complete=False)["status"]=="inconclusive"
    assert a.variance_band_status(*bounds,mean_Q30=0)["status"]=="inconclusive"


def test_zero_individual_Q_ratio_NA_does_not_poison_positive_mean_Q_primary():
    out=a.original_metrics(endpoints([[1,4,2,3],[4,1,3,2],[1,3,4,2],[4,2,1,3]]),1.,group_size=3,expected_repeats=4)
    assert out["original_comparisons"]["complete"][2]
    assert out["original_comparisons"]["Q30"][2]==0
    assert np.isnan(out["original_comparisons"]["ratio_Q120_Q30"][2])
    q30=np.r_[0.,np.linspace(1.,2.,29)];q120=q30.copy()
    l=a.mean_t_interval(q120-.8*q30);u=a.mean_t_interval(q120-1.25*q30)
    assert a.variance_band_status(l["low"],l["high"],u["low"],u["high"],q30.mean(),complete=l["valid"] and u["valid"])["status"]=="supported"
    assert a.joint_status(["inconclusive","not_supported"])=="not_supported"
    assert a.joint_status(["supported","supported"])=="supported"


def test_real_inputs_exact_60_hashes_readonly_M07_M06_M05_no_generation(monkeypatch):
    monkeypatch.setattr(a,"rng_for_panel",lambda *args,**kw:pytest.fail("original gen"))
    monkeypatch.setattr(finite_k,"rng_for_panel",lambda *args,**kw:pytest.fail("old generation"))
    inputs=a.make_inputs_manifest();cells=a.load_inputs(inputs)
    assert inputs["accepted_head"]==a.ACCEPTED_HEAD
    assert len(inputs["source_files_sha256"])==len(inputs["m06_files_sha256"])==len(inputs["m05_files_sha256"])==7
    assert inputs["originals_per_cell"]==10 and inputs["selected_originals"]==60 and "R" not in inputs
    assert "/Users/" not in json.dumps(inputs)
    for selection in inputs["selections"]:
        cell=cells[(selection["distribution"],selection["master_seed"])]
        assert cell["samples"].shape==(10,20)
        np.testing.assert_array_equal(cell["outer_id"],np.arange(10))
        assert selection["selected_samples_sha256"]==a.sample_sha256(cell["samples"])
        assert all(not x.flags.writeable for x in cell.values())
    changed=deepcopy(inputs);changed["selections"][0]["outer_ids"][-1]=10
    with pytest.raises(ValueError):a.validate_inputs_manifest(changed)


@pytest.mark.parametrize("drift",["bytes","metadata","lineage"])
def test_input_drift_stops_no_repair_or_endpoints_reuse(monkeypatch,drift):
    path="evidence/rq_m07/panels/gamma_seed0.npz"
    blobs={key:(a.ROOT/key).read_bytes() for key in a.INPUT_PATHS}
    if drift=="lineage":
        source=json.loads(blobs[a.M07_MANIFEST_PATH]);source["inputs"]["selected_originals"]=61
        blobs[a.M07_MANIFEST_PATH]=json.dumps(source).encode()
    else:
        with np.load(BytesIO(blobs[path]),allow_pickle=False) as panel:arrays={k:panel[k] for k in panel.files}
        if drift=="bytes":arrays["samples"][0,0]+=1
        else:arrays["namespace"]=np.array(a.MAIN_NAMESPACE)
        stream=BytesIO();np.savez_compressed(stream,**arrays);blobs[path]=stream.getvalue()
    monkeypatch.setattr(a,"_verified_blob",lambda root,key:blobs[key])
    with pytest.raises(ValueError):a.make_inputs_manifest()


def test_Git_blob_byte_drift_is_rejected(monkeypatch):
    actual=Path.read_bytes
    def changed(path):
        payload=actual(path)
        return payload+b" " if path==a.ROOT/a.M07_MANIFEST_PATH else payload
    monkeypatch.setattr(Path,"read_bytes",changed)
    with pytest.raises(ValueError,match="differs from accepted Git blob"):
        a.make_inputs_manifest()
