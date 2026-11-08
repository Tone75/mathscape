"""Deterministic RQ-M03 references, never a main panel or all-cell smoke run."""

import numpy as np
import pytest
from scipy import stats

from mathscape.bootstrap import intervals as m02_intervals
from mathscape.stability import (
    MAIN_NAMESPACE, SMOKE_NAMESPACE, METHODS, bootstrap_means,
    generate_bootstrap_means, generate_sample, infer_prefixes, mean_t_interval,
    original_metrics, prepare_sample, quantity_status, rng_for_panel, welch_interval,
)


SAMPLE = np.array([0., 1., 2., 3., 9.])


def compare_records(first, second, keys=("lower", "upper", "valid", "reason", "z0")):
    for method in METHODS:
        for key in keys:
            if isinstance(first[method][key], float):
                assert first[method][key] == pytest.approx(second[method][key], abs=1e-12, nan_ok=True)
            else:
                assert first[method][key] == second[method][key]


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_shared_small_prefixes_match_unchanged_m02_engine(alpha):
    means = np.linspace(-2, 8, 999)
    actual = infer_prefixes(prepare_sample(SAMPLE, alpha), means, (199, 999))
    for b in (199, 999):
        reference = m02_intervals(SAMPLE, means[:b], alpha)
        compare_records(actual[b], reference)
        assert actual[b]["bca"]["a"] == pytest.approx(reference["bca"]["a"])


def test_scipy_public_api_same_4999_distribution_reference():
    reference = stats.bootstrap((SAMPLE,), np.mean, n_resamples=4999, method="percentile",
                                confidence_level=.95, rng=np.random.default_rng(19))
    bca_reference = stats.bootstrap((SAMPLE,), np.mean, n_resamples=0, method="BCa",
                                    confidence_level=.95, bootstrap_result=reference)
    actual = infer_prefixes(prepare_sample(SAMPLE), reference.bootstrap_distribution)[4999]
    for method, ref in (("percentile", reference), ("bca", bca_reference)):
        assert actual[method]["valid"]
        assert [actual[method]["lower"], actual[method]["upper"]] == pytest.approx(
            [ref.confidence_interval.low, ref.confidence_interval.high], abs=1e-12)


def test_t_and_acceleration_prepared_once_invariant_across_k_and_b(monkeypatch):
    from mathscape import stability
    original = stability.t_inference
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(stability, "t_inference", counted)
    context = prepare_sample(SAMPLE)
    for shift in (0., .1, -.2):
        records = infer_prefixes(context, np.linspace(-2, 8, 999) + shift, (199, 999))
        for b in records:
            assert records[b]["t"] == context["t"]
            for method in METHODS:
                assert records[b][method]["a"] == context["a"]
    assert len(calls) == 1
    jackknife = np.array([np.mean(np.delete(SAMPLE, i)) for i in range(len(SAMPLE))])
    deviations = jackknife.mean() - jackknife
    assert context["a"] == pytest.approx(np.sum(deviations**3)/(6*np.sum(deviations**2)**1.5), abs=1e-15)


def test_prefix_specific_bias_correction_and_order_independence():
    means = np.r_[np.linspace(-2, 6, 199), np.linspace(1, 8, 800), np.linspace(2, 10, 4000)]
    context = prepare_sample(SAMPLE)
    first = infer_prefixes(context, means)
    reversed_order = infer_prefixes(context, means, (4999, 199, 999))
    z0s = []
    for b in (199, 999, 4999):
        prefix = means[:b]
        probability = ((prefix < SAMPLE.mean()).sum() + .5*(prefix == SAMPLE.mean()).sum())/b
        expected_z0 = stats.norm.ppf(probability)
        assert first[b]["bca"]["z0"] == pytest.approx(expected_z0)
        z0s.append(first[b]["bca"]["z0"])
        compare_records(first[b], reversed_order[b])
    assert len(set(z0s)) == 3


def test_tau_uses_min_at_each_endpoint_and_never_clips_quantiles():
    means = np.r_[np.linspace(-10, 2, 190), np.linspace(3, 5, 9)]
    result = infer_prefixes(prepare_sample(SAMPLE), means, (199,))[199]
    for method in ("percentile", "bca"):
        record = result[method]
        assert record["valid"]
        for suffix in ("low", "high"):
            q = record[f"q_{suffix}"]
            assert record[f"tau_{suffix}"] == pytest.approx(199*min(q,1-q))
        assert record["tail_warning"] == (min(record["tau_low"],record["tau_high"]) < 10)
        expected = np.quantile(means,[record["q_low"],record["q_high"]],method="linear")
        assert [record["lower"],record["upper"]] == pytest.approx(expected)


def test_positive_affine_equivariance_and_standardized_a_invariance():
    means = np.arange(-8,33)/4
    one = infer_prefixes(prepare_sample(SAMPLE), means, (19, 41))
    two = infer_prefixes(prepare_sample(8 + 2*SAMPLE), 8 + 2*means, (19, 41))
    for b in one:
        for method in METHODS:
            assert one[b][method]["valid"] == two[b][method]["valid"]
            if one[b][method]["valid"]:
                assert two[b][method]["lower"] == pytest.approx(8+2*one[b][method]["lower"])
                assert two[b][method]["upper"] == pytest.approx(8+2*one[b][method]["upper"])
                assert two[b][method]["a"] == pytest.approx(one[b][method]["a"])


def test_nonfinite_unused_suffix_does_not_invalidate_finite_prefix():
    means = np.linspace(-2,8,99)
    means[-1] = np.nan
    records = infer_prefixes(prepare_sample(SAMPLE), means, (19,99))
    reference = infer_prefixes(prepare_sample(SAMPLE), means[:19], (19,))
    compare_records(records[19],reference[19])
    assert records[99]["t"]["valid"]
    for method in ("percentile","bca"):
        assert not records[99][method]["valid"]
        assert records[99][method]["reason"] == "bootstrap_distribution_nonfinite"


@pytest.mark.parametrize("sample", [[1], [2,2,2], [1,np.nan,3], [1,np.inf,3], [[1,2]], [True,False], [1j,2j]])
def test_invalid_original_reasons_without_silent_fallback(sample):
    context=prepare_sample(sample)
    assert not context["valid"] and context["reason"]
    record=infer_prefixes(context,np.arange(19), (19,))[19]
    for method in METHODS:
        assert not record[method]["valid"]
        assert record[method]["reason"]
        assert np.isnan(record[method]["lower"])


def test_nonfinite_z0_keeps_t_percentile_and_invalidates_bca_only():
    record=infer_prefixes(prepare_sample(SAMPLE),np.full(19,-1.),(19,))[19]
    assert record["t"]["valid"] and record["percentile"]["valid"]
    assert not record["bca"]["valid"] and record["bca"]["reason"] == "bca_bias_correction_nonfinite"


def test_rng_exact_namespace_outer_purpose_inner_mapping():
    assert MAIN_NAMESPACE==0x4D3033 and SMOKE_NAMESPACE==0x534D3033
    actual=rng_for_panel("centered_exponential",5,2,7,1,3,SMOKE_NAMESPACE)
    expected=np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        entropy=2,spawn_key=(SMOKE_NAMESPACE,2,5,7,1,3))))
    np.testing.assert_array_equal(actual.bit_generator.random_raw(20),expected.bit_generator.random_raw(20))
    original_rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        entropy=2,spawn_key=(SMOKE_NAMESPACE,2,5,7,0,0))))
    np.testing.assert_array_equal(generate_sample("centered_exponential",5,2,7,SMOKE_NAMESPACE),
                                  original_rng.exponential(1,5)-1)


def test_draw_order_inner_separation_prefix_and_batch_invariance():
    sample=generate_sample("normal",5,1,2,SMOKE_NAMESPACE)
    first=generate_bootstrap_means(sample,"normal",5,1,2,0,99,SMOKE_NAMESPACE)
    other=generate_bootstrap_means(sample,"normal",5,1,2,1,99,SMOKE_NAMESPACE)
    assert not np.array_equal(first,other)
    generate_sample("centered_exponential",20,2,3,SMOKE_NAMESPACE)
    short=generate_bootstrap_means(sample,"normal",5,1,2,0,19,SMOKE_NAMESPACE,max_elements=5)
    np.testing.assert_array_equal(short,first[:19])
    for cap in (5,15,1000000):
        actual=bootstrap_means(SAMPLE,np.random.default_rng(42),99,max_elements=cap)
        expected=bootstrap_means(SAMPLE,np.random.default_rng(42),99)
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(sample,generate_sample("normal",5,1,2,SMOKE_NAMESPACE))


def fixture_data(K=20):
    lower=np.zeros((3,3,K));upper=np.zeros_like(lower)
    lower[:,0,:]=-1;upper[:,0,:]=1
    for bi in range(3):
        for mi in (1,2):
            lower[bi,mi]=np.linspace(-2,1,K)/(bi+1)
            upper[bi,mi]=lower[bi,mi]+2
    return {"lower":lower,"upper":upper,"valid":np.ones_like(lower,dtype=bool),
            "tail_warning":np.zeros_like(lower,dtype=bool)}


def test_original_unit_metrics_ddof_one_and_t_single_control():
    data=fixture_data();actual=original_metrics(data,.5)
    assert actual["complete"].all()
    assert (actual["total_inner"][:,0]==1).all()
    assert (actual["total_inner"][:,1:]==20).all()
    assert (actual["A"][:,0]==0).all() and (actual["lower_sd"][:,0]==0).all()
    lo,hi=data["lower"][0,2],data["upper"][0,2]
    expected=np.sqrt((lo.var(ddof=1)+hi.var(ddof=1))/2)/.5
    assert actual["A"][0,2]==pytest.approx(expected)
    assert actual["C"][0,2]==np.mean((lo<=0)&(0<=hi))
    assert actual["mean_width"][0,2]==pytest.approx(np.mean(hi-lo))
    np.testing.assert_allclose(actual["C"]+actual["miss_above"]+actual["miss_below"],1)


def test_any_inner_invalid_makes_original_primary_metrics_na_with_counts():
    data=fixture_data();data["valid"][1,2,3]=False
    actual=original_metrics(data,.5)
    assert not actual["complete"][1,2]
    assert actual["valid_inner"][1,2]==19 and actual["invalid_inner"][1,2]==1
    for key in ("A","C","lower_sd","upper_sd","mean_width","warning_rate","miss_above","miss_below"):
        assert np.isnan(actual[key][1,2])
    assert actual["complete"][0,2]


def test_t_control_cannot_be_changed_by_inner_stream():
    data=fixture_data();data["lower"][1,0,3]=-.9
    with pytest.raises(ValueError,match="t control"):
        original_metrics(data,.5)


@pytest.mark.parametrize("se", [0,-1,np.nan,np.inf,True])
def test_bad_standardization_scale_invalidates_primary_metrics(se):
    actual=original_metrics(fixture_data(),se)
    assert not actual["complete"].any()
    assert np.isnan(actual["A"]).all() and np.isnan(actual["C"]).all()
    assert (actual["valid_inner"][:,1:]==20).all()


def test_h2_classifier_and_evaluation_use_separate_inner_halves():
    data=fixture_data();data["tail_warning"][1,2,:5]=True
    before=original_metrics({k:v[:,:,10:] for k,v in data.items()},.5)
    classifier=(data["tail_warning"][1,2,:10].sum()>=5)
    assert classifier
    # A large alteration confined to classification endpoints cannot alter eval A.
    data["lower"][1,2,:10]-=50;data["upper"][1,2,:10]+=80
    after=original_metrics({k:v[:,:,10:] for k,v in data.items()},.5)
    np.testing.assert_array_equal(before["A"],after["A"])
    data["tail_warning"][1,2,10:]=True
    assert (data["tail_warning"][1,2,:10].sum()>=5)==classifier


def test_mean_t_interval_uses_original_units_and_not_K_pseudoreplication():
    values=np.array([.1,.2,.3,.7])
    actual=mean_t_interval(values)
    mean=values.mean();sd=values.std(ddof=1);se=sd/np.sqrt(4);critical=stats.t.ppf(.995,3)
    assert actual["N"]==4 and actual["df"]==3 and actual["valid"]
    assert [actual["mean"],actual["sd"],actual["mcse"]]==pytest.approx([mean,sd,se])
    assert [actual["low"],actual["high"]]==pytest.approx([mean-critical*se,mean+critical*se])
    assert actual["mcse"] != pytest.approx(sd/np.sqrt(4*20))


def test_original_mean_interval_is_not_clipped_to_probability_domain():
    result=mean_t_interval([.8,.9,1.])
    assert result["valid"] and result["high"]>1


def test_nan_original_is_not_removed_from_mean_denominator():
    result=mean_t_interval([.1,np.nan,.3])
    assert result["N"]==3 and result["finite_count"]==2
    assert not result["valid"] and np.isnan(result["mean"]) and np.isnan(result["mcse"])


@pytest.mark.parametrize("values", [[],[1],[0,0,0]])
def test_too_few_or_zero_variance_confirmatory_mean_is_inconclusive(values):
    result=mean_t_interval(values)
    assert not result["valid"] and np.isnan(result["low"])
    if len(values)>1:
        assert result["mean"]==0 and result["sd"]==0 and result["mcse"]==0


def test_welch_matches_independent_scipy_public_reference():
    one=np.array([.3,.5,.8,1.1,1.3]);zero=np.array([.1,.2,.4,.5,.7,.9,.95])
    actual=welch_interval(one,zero)
    ref=stats.ttest_ind(one,zero,equal_var=False)
    ci=ref.confidence_interval(confidence_level=.99)
    assert actual["valid"] and (actual["N1"],actual["N0"])==(5,7)
    assert actual["difference"]==pytest.approx(one.mean()-zero.mean())
    assert actual["df"]==pytest.approx(ref.df)
    assert actual["se"]==pytest.approx(np.sqrt(one.var(ddof=1)/5+zero.var(ddof=1)/7))
    assert [actual["low"],actual["high"]]==pytest.approx([ci.low,ci.high],abs=1e-12)


@pytest.mark.parametrize("minimum", [5,30])
def test_h2_group_count_gate_is_enforced_before_interval_decision(minimum):
    result=welch_interval(np.arange(minimum-1),np.arange(minimum)+100,min_group=minimum)
    assert not result["valid"] and result["reason"]=="insufficient_group_count"
    assert np.isnan(result["low"]) and np.isnan(result["high"])
    assert welch_interval(np.arange(minimum),np.arange(minimum)+100,min_group=minimum)["valid"]


def test_welch_invalids_are_not_excluded_and_both_constant_is_undefined():
    result=welch_interval([.1,np.nan,.3],[.1,.2,.3])
    assert result["N1"]==3 and result["finite_count1"]==2
    assert not result["valid"] and np.isnan(result["difference"])
    result=welch_interval([1,1,1],[2,2,2])
    assert not result["valid"] and result["reason"]=="variance_zero_or_nonfinite"


def test_exact_constant_float_vectors_do_not_get_false_nonzero_variance():
    # A naive reduction can give positive roundoff SD for np.full(300,.8).
    result=mean_t_interval(np.full(300,.8))
    assert not result["valid"] and result["sd"]==0 and result["mcse"]==0
    assert result["mean"]==.8 and np.isnan(result["low"])
    result=welch_interval(np.full(300,.8),np.full(300,.7),min_group=30)
    assert not result["valid"] and result["variance1"]==result["variance0"]==0


@pytest.mark.parametrize("low,high,threshold,direction,seeds,expected", [
    (-.5,-.1,0,"less",[-.1,-.2,-.3],"supported"),
    (-.5,0,0,"less",[-.1,-.2,-.3],"inconclusive"),
    (-.5,-.1,0,"less",[-.1,.2,-.3],"inconclusive"),
    (.1,.5,0,"less",[-.1,np.nan,-.3],"not_supported"),
    (.03,.05,.02,"greater",[.03,.04,.05],"supported"),
    (.01,.015,.02,"greater",[.03,.04,.05],"not_supported"),
    (.02,.05,.02,"greater",[.03,.04,.05],"inconclusive"),
    (np.nan,np.nan,.02,"greater",[.03,.04,.05],"inconclusive"),
])
def test_exact_strict_threshold_seed_conditions_and_clear_opposite_priority(low,high,threshold,direction,seeds,expected):
    assert quantity_status(low,high,threshold,direction,seeds)==expected


@pytest.mark.parametrize("key,value", [("n",1),("n",201),("outer_id",-1),("outer_id",100),
    ("inner_id",20),("purpose_id",2),("master_seed",-1),("master_seed",True),
    ("namespace",2**32),("namespace",np.nan)])
def test_invalid_rng_arguments(key,value):
    kwargs={"distribution":"normal","n":5,"namespace":SMOKE_NAMESPACE}|{key:value}
    with pytest.raises(ValueError):rng_for_panel(**kwargs)


def test_original_rng_fixed_inner_id_and_resample_shape_limits():
    with pytest.raises(ValueError):rng_for_panel("normal",5,inner_id=1,purpose_id=0)
    for cap in (4,1000001,True):
        with pytest.raises(ValueError):bootstrap_means(SAMPLE,np.random.default_rng(0),99,cap)
    with pytest.raises(ValueError):bootstrap_means(SAMPLE,np.random.default_rng(0),5000)
    with pytest.raises(ValueError):generate_bootstrap_means(SAMPLE,"normal",20,namespace=SMOKE_NAMESPACE)


@pytest.mark.parametrize("B_values", [[],[19,19],[0],[5000],[19.5]])
def test_invalid_prefixes(B_values):
    with pytest.raises(ValueError):infer_prefixes(prepare_sample(SAMPLE),np.arange(99),B_values)


def test_prefix_cannot_exceed_saved_series_length():
    with pytest.raises(ValueError):infer_prefixes(prepare_sample(SAMPLE),np.arange(99),(199,))
