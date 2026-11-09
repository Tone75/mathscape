"""Deterministic finite-K arithmetic and saved-input contracts, no experiment."""

from copy import deepcopy
from io import BytesIO
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from mathscape import finite_k as fk
from mathscape import matched_moments, matched_stability, stability


SAMPLE = np.array([0., .1, .2, .2, .3, .5, .7, .9, 1., 1.2,
                   1.5, 1.8, 2., 2.5, 3., 3.2, 4., 5., 7., 9.])


def endpoint_fixture(j=2, group_size=2):
    # Four groups with endpoint variances2,8,18,32 and means1,12,23,34.
    base = np.array([0, 2, 10, 14, 20, 26, 30, 38], float)
    if group_size != 2:
        base = np.concatenate([np.linspace(11*g,11*g+2*(g+1),group_size) for g in range(4)])
    lower = np.empty((j,3,4*group_size))
    upper = np.empty_like(lower)
    for r in range(j):
        for method in (1,2):
            lower[r,method]=base*(1+.1*r+.03*method)-10
            upper[r,method]=lower[r,method]+3+.1*method
    lower[:,0],upper[:,0]=.8,1.8
    return dict(lower=lower,upper=upper,valid=np.ones(lower.shape,bool),
                tail_warning=np.zeros(lower.shape,bool))


def one_replicate(data,r=0):
    return {key:value[r] for key,value in data.items()}


def test_rng_exact_spawn_key_namespaces_and_no_original_generator():
    assert fk.DISTRIBUTION_IDS=={"gamma":1,"lognormal":2}
    assert fk.MAIN_NAMESPACE==0x4D3037 and fk.SMOKE_NAMESPACE==0x534D3037
    assert not hasattr(fk,"generate_sample")
    for dist,identifier in fk.DISTRIBUTION_IDS.items():
        actual=fk.rng_for_panel(dist,20,2,9,1,19,119)
        expected=np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            entropy=2,spawn_key=(0x4D3037,identifier,20,9,1,19,119))))
        np.testing.assert_array_equal(actual.bit_generator.random_raw(5),expected.bit_generator.random_raw(5))
    assert fk.MAIN_NAMESPACE not in (matched_moments.MAIN_NAMESPACE,matched_stability.MAIN_NAMESPACE)
    with pytest.raises(ValueError):fk.rng_for_panel("gamma",purpose_id=0)


@pytest.mark.parametrize("kwargs",[dict(distribution="normal"),dict(distribution=[]),dict(n=19),
    dict(n=True),dict(master_seed=3),dict(master_seed=-1),dict(master_seed=np.nan),
    dict(outer_id=10),dict(outer_id=True),dict(replicate_id=20),dict(replicate_id=-1),
    dict(inner_id=120),dict(inner_id=1.),dict(namespace=matched_stability.MAIN_NAMESPACE)])
def test_bad_rng_keys_reject(kwargs):
    with pytest.raises(ValueError):fk.rng_for_panel(**({"distribution":"gamma"}|kwargs))


def test_tiny_fixed_sample_int64_index_reference_batch_and_order_invariance():
    arguments=dict(distribution="lognormal",master_seed=1,outer_id=7,replicate_id=3,
                   inner_id=29,Bmax=17,namespace=fk.SMOKE_NAMESPACE)
    means,indices=fk.generate_bootstrap_means(SAMPLE,**arguments,return_indices=True)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=1,
        spawn_key=(fk.SMOKE_NAMESPACE,2,20,7,1,3,29))))
    expected=rng.integers(0,20,size=(17,20),dtype=np.int64)
    assert indices.dtype==np.uint8
    np.testing.assert_array_equal(indices.astype(np.int64),expected)
    np.testing.assert_array_equal(means,SAMPLE[expected].mean(axis=1))
    fk.rng_for_panel("gamma",replicate_id=8,inner_id=11)
    for limit in (20,60,1_000_000):
        np.testing.assert_array_equal(fk.generate_bootstrap_means(SAMPLE,**arguments,max_elements=limit),means)
    changed=fk.generate_bootstrap_means(SAMPLE,**(arguments|dict(replicate_id=4)))
    assert not np.array_equal(changed,means)


def test_generation_dtype_and_memory_bound_without_recomputing_t(monkeypatch):
    calls=[]
    class Fixed:
        def integers(self,low,high,size,dtype):
            assert (low,high,dtype)==(0,20,np.int64)
            calls.append(size)
            return np.arange(np.prod(size),dtype=np.int64).reshape(size)%20
    monkeypatch.setattr(fk,"rng_for_panel",lambda *a,**k:Fixed())
    monkeypatch.setattr(fk,"prepare_sample",lambda *a,**k:pytest.fail("t/a recomputed"))
    means,indices=fk.generate_bootstrap_means(SAMPLE,"gamma",Bmax=7,max_elements=60,return_indices=True)
    assert calls==[(3,20),(3,20),(1,20)]
    np.testing.assert_array_equal(means,SAMPLE[indices].mean(axis=1))


@pytest.mark.parametrize("sample,kwargs",[(np.zeros(20),{}),(np.full(20,np.nan),{}),
    (np.ones(19),{}),(np.ones((2,10)),{}),(np.arange(20,dtype=complex),{}),
    (SAMPLE,{"Bmax":5000}),(SAMPLE,{"max_elements":19}),
    (SAMPLE,{"max_elements":1000001}),(SAMPLE,{"return_indices":1})])
def test_bad_sample_or_resources_reject(sample,kwargs):
    with pytest.raises(ValueError):fk.generate_bootstrap_means(sample,"gamma",**kwargs)


def test_inference_reuses_shared_distribution_and_public_scipy_reference():
    reference=stats.bootstrap((SAMPLE,),np.mean,n_resamples=4999,method="percentile",
                             confidence_level=.95,rng=np.random.default_rng(74))
    bca=stats.bootstrap((SAMPLE,),np.mean,n_resamples=0,method="BCa",
                       confidence_level=.95,bootstrap_result=reference)
    context=fk.prepare_sample(SAMPLE)
    actual=fk.infer_single(context,reference.bootstrap_distribution)
    assert fk.prepare_sample is stability.prepare_sample
    assert set(inspect.signature(fk.infer_single).parameters)=={"context","means"}
    assert actual["t"]==context["t"]
    for name,result in [("percentile",reference),("bca",bca)]:
        assert [actual[name]["lower"],actual[name]["upper"]]==pytest.approx(
            [result.confidence_interval.low,result.confidence_interval.high],abs=1e-12)
    tie=fk.infer_single(fk.prepare_sample([0.,1.,2.],alpha=.5),[0.,1.,1.,2.])
    assert tie["bca"]["z0"]==0 and tie["bca"]["a"]==0
    assert tie["percentile"]["lower"]==.75


def test_hand_reference_four_groups_jensen_and_endpoint_sum_of_squares():
    lo=np.array([0,2,10,14,20,26,30,38],float)
    data=dict(lower=np.stack([np.full(8,.8),lo,lo]),
              upper=np.stack([np.full(8,1.8),lo+3,lo+3]),
              valid=np.ones((3,8),bool),tail_warning=np.zeros((3,8),bool))
    out=fk.replicate_metrics(data,1.,group_size=2)
    np.testing.assert_allclose(out["group_lower_var"][2],[2,8,18,32])
    np.testing.assert_allclose(out["group_lower_mean"][2],[1,12,23,34])
    assert out["V30"][2]==2
    assert out["V120"][2]==pytest.approx(1270/7)
    assert out["Vbar"][2]==15
    assert out["Abar"][2]==pytest.approx(2.5*np.sqrt(2))
    assert out["Jgap"][2]==pytest.approx(np.sqrt(15)-2.5*np.sqrt(2))
    assert out["Pgap"][2]==pytest.approx(np.sqrt(1270/7)-np.sqrt(15))
    assert out["Rgap"][2]==pytest.approx(1.5*np.sqrt(2))
    assert out["delta"][2]==pytest.approx(np.sqrt(1270/7)-np.sqrt(2))
    assert out["Jgap"][2]>0 and out["algebra_ok"].all()
    assert out["lower_ss_scale"][2]==pytest.approx(1270)
    assert abs(out["lower_ss_residual"][2])<fk.IDENTITY_ATOL
    assert abs(out["decomposition_residual"][2])<fk.IDENTITY_ATOL
    np.testing.assert_allclose(out["group_V"][2],out["group_A"][2]**2)
    assert out["by_K"]["Ase"][0,2]==out["A30"][2]
    assert out["by_K"]["Vse"][1,2]==pytest.approx(out["V120"][2])


def test_main_sized_deterministic_fixture_tolerance_lock_and_no_gap_clipping():
    data=endpoint_fixture(j=20,group_size=30)
    out=fk.original_metrics(data,.5)
    rep=out["replicate"]
    assert out["original"]["complete"].all() and rep["algebra_ok"].all()
    assert fk.NUMERIC_RTOL==2e-10 and fk.NUMERIC_ATOL==2e-12
    assert fk.IDENTITY_RTOL==fk.IDENTITY_ATOL==2e-12
    for key in ("decomposition_abs_normalized_residual","lower_ss_abs_normalized_residual","upper_ss_abs_normalized_residual"):
        assert np.max(rep[key])<fk.IDENTITY_ATOL
    assert np.min(rep["Jgap"])>=-fk.IDENTITY_ATOL
    np.testing.assert_allclose(rep["delta"],rep["Jgap"]+rep["Pgap"]+rep["Rgap"],rtol=fk.IDENTITY_RTOL,atol=fk.IDENTITY_ATOL)
    assert rep["by_K"]["Ase"].shape==(20,2,3)


def test_original_paired_difference_repeated_variances_and_ratios_reference():
    data=endpoint_fixture(j=5)
    out=fk.original_metrics(data,.5,group_size=2,expected_replicates=5)
    rep,original=out["replicate"],out["original"]
    for mi in (1,2):
        first,full=rep["A30"][:,mi],rep["A120"][:,mi]
        assert original["d"][mi]==pytest.approx(np.mean(full-first))
        assert original["Q30"][mi]==pytest.approx(np.var(first,ddof=1))
        assert original["Q120"][mi]==pytest.approx(np.var(full,ddof=1))
        assert original["W"][mi]==pytest.approx(np.var(full,ddof=1)-.5*np.var(first,ddof=1))
        assert original["ratio_Q120_Q30"][mi]==pytest.approx(np.var(full,ddof=1)/np.var(first,ddof=1))
        np.testing.assert_allclose(out["by_K"]["Ase"][:,mi],rep["by_K"]["Ase"][:,:,mi].mean(axis=0))
        assert original["mean_A30"][mi]!=pytest.approx(rep["Abar"][:,mi].mean())
    assert original["planned_replicates"].tolist()==[5,5,5]
    assert original["total_inner"].tolist()==[1,40,40]


def test_t_all_variances_gaps_exact_zero_zero_ratio_na_and_ci_undefined():
    out=fk.original_metrics(endpoint_fixture(j=20,group_size=30),.5)
    for key in ("V30","A30","V120","A120","Vbar","Abar","Jgap","Pgap","Rgap","delta",
                "decomposition_residual","lower_ss_residual","upper_ss_residual"):
        assert np.all(out["replicate"][key][:,0]==0)
    for key in ("Q30","Q120","W","d","mean_A30","mean_A120"):
        assert out["original"][key][0]==0
    assert np.isnan(out["original"]["ratio_Q120_Q30"][0])
    assert not fk.mean_t_interval(np.full(30,.8))["valid"]
    assert not fk.mean_t_interval(np.zeros(30))["valid"]
    old=matched_moments.original_metrics(dict(lower=np.full(3,-1.),upper=np.full(3,1.),
        valid=np.ones(3,bool),tail_warning=np.zeros(3,bool)))
    assert all(np.isnan(old[key]).all() for key in ("lower_sd","upper_sd","S","Ase","Apop"))


def test_t_invalid_is_method_local_and_missing_replicate_padding_retains_plan():
    data=endpoint_fixture(j=3)
    data["valid"][1,0,4]=False
    data["lower"][1,0,4]=np.nan
    out=fk.original_metrics(data,.5,group_size=2,expected_replicates=3)
    assert out["original"]["complete"].tolist()==[False,True,True]
    assert np.isnan(out["original"]["d"][0]) and np.isfinite(out["original"]["d"][2])
    assert out["original"]["total_inner"][0]==1 and out["original"]["invalid_inner"][0]==1
    data=endpoint_fixture(j=3)
    data["valid"][2]=False
    data["lower"][2]=np.nan;data["upper"][2]=np.nan
    out=fk.original_metrics(data,.5,group_size=2,expected_replicates=3)
    assert out["original"]["planned_replicates"].tolist()==[3,3,3]
    assert out["original"]["complete_replicates"].tolist()==[2,2,2]
    assert not out["original"]["complete"].any()
    assert np.isnan(out["original"]["d"]).all()
    assert out["original"]["invalid_inner"].tolist()==[1,8,8]
    with pytest.raises(ValueError,match="planned replicate"):
        fk.original_metrics({k:v[:2] for k,v in data.items()},.5,group_size=2,expected_replicates=3)


def test_full_K_completeness_not_just_first_group_and_warning_preserved():
    data=endpoint_fixture(j=2)
    data["valid"][1,1,7]=False
    data["tail_warning"][:,2,0]=True
    out=fk.original_metrics(data,.5,group_size=2,expected_replicates=2)
    assert out["original"]["complete"].tolist()==[True,False,True]
    assert np.isnan(out["original"]["d"][1])
    assert np.isnan(out["by_K"]["Ase"][:,1]).all()
    assert out["by_K"]["warning_rate"][0,2]==.5
    assert out["by_K"]["warning_rate"][1,2]==.125
    assert out["original"]["total_inner"].tolist()==[1,16,16]
    np.testing.assert_allclose(out["by_K"]["C"][:,2]+out["by_K"]["Mbelow"][:,2]+out["by_K"]["Mabove"][:,2],1)


@pytest.mark.parametrize("se",[0,-1,np.nan,np.inf,True,"0.5",10**1000,1e-300])
def test_invalid_or_undefined_scale_never_zero_fills(se):
    result=fk.original_metrics(endpoint_fixture(),se,group_size=2,expected_replicates=2)
    assert not result["original"]["complete"].any()
    assert np.isnan(result["original"]["d"]).all()


def test_fixed_affine_equivariance_and_raw_decomposition():
    data=endpoint_fixture(j=3)
    first=fk.original_metrics(data,.5,group_size=2,expected_replicates=3)
    changed={k:v*2.5+3.7 if k in ("lower","upper") else v.copy() for k,v in data.items()}
    second=fk.original_metrics(changed,1.25,group_size=2,expected_replicates=3)
    for key in ("A30","A120","V30","V120","Jgap","Pgap","Rgap","delta"):
        np.testing.assert_allclose(first["replicate"][key],second["replicate"][key],rtol=2e-13,atol=2e-13)
    for key in ("d","Q30","Q120","W"):
        np.testing.assert_allclose(first["original"][key],second["original"][key],rtol=2e-13,atol=2e-13)


def test_changed_observed_t_endpoint_rejects_corruption():
    data=endpoint_fixture()
    data["lower"][1,0,0]=.7
    with pytest.raises(ValueError,match="valid observed t"):
        fk.original_metrics(data,.5,group_size=2,expected_replicates=2)


@pytest.mark.parametrize("bounds,status",[((-.002,.002),"supported"),
    ((.0026,.004),"not_supported"),((-.004,-.0026),"not_supported"),
    ((-.0025,.002),"inconclusive"),((-.002,.0025),"inconclusive"),
    ((.0025,.004),"inconclusive"),((-.004,-.0025),"inconclusive"),
    ((-.003,.003),"inconclusive"),((0,0),"inconclusive"),((np.nan,.002),"inconclusive")])
def test_H1_supports_inside_and_rejects_outside_opposite_previous_H1(bounds,status):
    assert fk.band_status(*bounds)["status"]==status
    assert fk.band_status(*bounds,complete=False)["status"]=="inconclusive"


@pytest.mark.parametrize("bounds,status",[((-.02,-.001),"supported"),((0,.1),"not_supported"),
    ((.01,.1),"not_supported"),((-.1,0),"inconclusive"),((0,0),"inconclusive"),
    ((np.nan,.1),"inconclusive")])
def test_H2_variance_reduction_boundary_and_SE0(bounds,status):
    assert fk.reduction_status(*bounds)["status"]==status
    assert fk.reduction_status(*bounds,complete=False)["status"]=="inconclusive"


def test_joint_status_and_original_unit_df29_not_JK_and_all_NA_strict():
    assert fk.joint_status(["supported","supported"])=="supported"
    assert fk.joint_status(["not_supported","inconclusive"])=="not_supported"
    assert fk.joint_status(["supported","inconclusive"])=="inconclusive"
    values=np.linspace(-.002,.002,30)
    actual=fk.mean_t_interval(values)
    assert actual["N"]==30 and actual["df"]==29
    assert actual["mcse"]==pytest.approx(np.std(values,ddof=1)/np.sqrt(30))
    reference=stats.ttest_1samp(values,0).confidence_interval(.99)
    assert [actual["low"],actual["high"]]==pytest.approx([reference.low,reference.high])
    values[0]=np.nan
    actual=fk.mean_t_interval(values)
    assert actual["N"]==30 and actual["finite_count"]==29 and not actual["valid"]


@pytest.fixture(scope="module")
def saved_blobs():
    # Only reading accepted saved evidence; no new original/random draw.
    paths=list(fk.INPUT_PATHS)+["evidence/rq_m05/manifest.json"]+[
        f"evidence/rq_m05/panels/{d}_seed{s}.npz" for d in fk.DISTRIBUTIONS for s in (0,1,2)]
    return {p:(fk.ROOT/p).read_bytes() for p in paths}


def test_verified_real_saved_inputs_M06_M05_correspondence_readonly_no_rng(monkeypatch):
    monkeypatch.setattr(fk,"rng_for_panel",lambda *a,**k:pytest.fail("original generation requested"))
    inputs=fk.make_inputs_manifest();cells=fk.load_inputs(inputs)
    assert inputs["accepted_head"]=="e2feae23397d014f6aa3363773da9913da85fcf7"
    assert len(inputs["source_files_sha256"])==len(inputs["upstream_files_sha256"])==7
    assert inputs["selected_originals"]==60 and len(cells)==6
    assert "/Users/" not in json.dumps(inputs)
    for selection in inputs["selections"]:
        cell=cells[(selection["distribution"],selection["master_seed"])]
        assert cell["samples"].shape==(10,20)
        np.testing.assert_array_equal(cell["outer_id"],np.arange(10))
        assert selection["outer_ids"]==list(range(10))
        assert selection["selected_samples_sha256"]==fk.sample_sha256(cell["samples"])
        assert all(not a.flags.writeable for a in cell.values())


def test_input_manifest_selection_drift_rejects_without_repair(monkeypatch,saved_blobs):
    monkeypatch.setattr(fk,"_verified_blob",lambda root,path:saved_blobs[path])
    inputs=fk.make_inputs_manifest()
    changed=deepcopy(inputs);changed["selections"][0]["outer_ids"][-1]=10
    for operation in (fk.load_inputs,fk.validate_inputs_manifest):
        with pytest.raises(ValueError,match="differs"):
            operation(changed)


@pytest.mark.parametrize("drift",["M06hash","M06ids","M06metadata","M05bytes","inputprovenance"])
def test_saved_input_drift_stops_not_generates(monkeypatch,saved_blobs,drift):
    blobs=dict(saved_blobs)
    path="evidence/rq_m06/panels/gamma_seed0.npz"
    if drift=="inputprovenance":
        source=json.loads(blobs[fk.M06_MANIFEST_PATH]);source["inputs"]["selections"][0]["outer_ids"][-1]=200
        blobs[fk.M06_MANIFEST_PATH]=json.dumps(source).encode()
    else:
        if drift=="M05bytes":path="evidence/rq_m05/panels/gamma_seed0.npz"
        with np.load(BytesIO(blobs[path]),allow_pickle=False) as p:arrays={k:p[k] for k in p.files}
        if drift=="M06hash":arrays["samples"][199,0]+=1
        elif drift=="M06ids":arrays["outer_id"][[0,1]]=arrays["outer_id"][[1,0]]
        elif drift=="M06metadata":arrays["namespace"]=np.array(fk.MAIN_NAMESPACE)
        else:arrays["samples"][0,0]+=1
        stream=BytesIO();np.savez_compressed(stream,**arrays);blobs[path]=stream.getvalue()
    monkeypatch.setattr(fk,"_verified_blob",lambda root,path:blobs[path])
    monkeypatch.setattr(fk,"rng_for_panel",lambda *a,**k:pytest.fail("drift repaired by generation"))
    with pytest.raises(ValueError):fk.make_inputs_manifest()


def test_full_local_bytes_must_match_accepted_Git_blob(monkeypatch):
    real_read=Path.read_bytes
    def changed(path):
        payload=real_read(path)
        return payload+b" " if path==fk.ROOT/fk.M06_MANIFEST_PATH else payload
    monkeypatch.setattr(Path,"read_bytes",changed)
    with pytest.raises(ValueError,match="differs from accepted Git blob"):
        fk.make_inputs_manifest()
