"""Analytic/fixed references only: no population experiment or smoke panel."""

import inspect

import numpy as np
import pytest
from scipy import stats

from mathscape import matched_moments as matched
from mathscape import stability


SAMPLE = np.array([0., .1, .2, .2, .3, .5, .7, .9, 1., 1.2,
                   1.5, 1.8, 2., 2.5, 3., 3.2, 4., 5., 7., 9.])


def test_analytic_lognormal_constants_cubic_and_recorded_values():
    v, t, s, m, d = (matched.LOGNORMAL_CONSTANTS[key] for key in ("v", "t", "s", "m", "d"))
    assert v**3 + 3*v == pytest.approx(2, abs=1e-14)
    assert t == 1+v**2
    assert s == pytest.approx(np.sqrt(np.log1p(v*v)), abs=1e-15)
    assert (t+2)*np.sqrt(t-1) == pytest.approx(2, abs=1e-14)
    assert s == pytest.approx(.5513835899408821, abs=1e-15)
    assert m == pytest.approx(1.1641741268419084, abs=1e-15)
    assert d == pytest.approx(.6939311786844594, abs=1e-15)
    assert matched.LOGNORMAL_EXCESS_KURTOSIS == pytest.approx(7.863462451016, abs=1e-12)
    assert matched.LOGNORMAL_LOWER_BOUND == pytest.approx(-1.677650698804, abs=1e-12)


def test_public_scipy_mvsk_and_support_preflight_all_absolute_errors_within_1e_minus10():
    checks = matched.validate_population_moments()
    for distribution in matched.DISTRIBUTIONS:
        check = checks[distribution]
        assert check["passed"] and check["tolerance"] == 1e-10
        assert all(error <= 1e-10 for error in check["absolute_errors"].values())
        actual = check["actual"]
        assert [actual[key] for key in ("mean", "variance", "skewness")] == pytest.approx([0, 1, 2], abs=1e-10)
    assert checks["gamma"]["actual"]["excess_kurtosis"] == 6
    assert checks["gamma"]["actual"]["lower_bound"] == -1
    assert checks["lognormal"]["actual"]["excess_kurtosis"] != 6
    assert checks["lognormal"]["actual"]["lower_bound"] != -1


def test_pdf_change_of_variables_and_gamma_exponential_identity():
    x = np.array([-.9, -.5, 0., .5, 2.])
    gamma = matched.population_distribution("gamma")
    np.testing.assert_allclose(gamma.pdf(x), stats.gamma.pdf(x+1, a=1, loc=0, scale=1))
    np.testing.assert_allclose(gamma.pdf(x), stats.expon.pdf(x, loc=-1, scale=1))
    lognormal = matched.population_distribution("lognormal")
    expected = matched.LOGNORMAL_D * stats.lognorm.pdf(
        matched.LOGNORMAL_D*x+matched.LOGNORMAL_M, s=matched.LOGNORMAL_S, loc=0, scale=1)
    np.testing.assert_allclose(lognormal.pdf(x), expected, rtol=1e-13)


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf, True, "1e-10", 10**1000])
def test_bad_moment_tolerance_rejected(tolerance):
    with pytest.raises(ValueError):
        matched.validate_population_moments(tolerance)


def test_too_strict_numerical_moment_check_fails_explicitly():
    with pytest.raises(ValueError, match="moment check failed"):
        matched.validate_population_moments(1e-16)


def test_generation_uses_population_transform_without_sample_normalization(monkeypatch):
    raw = np.linspace(.1, 3.8, 20)
    z = np.linspace(-2, 2.5, 20)
    class FixedGenerator:
        def gamma(self, shape, scale, size):
            assert (shape, scale, size) == (1., 1., 20)
            return raw.copy()
        def standard_normal(self, size):
            assert size == 20
            return z.copy()
    monkeypatch.setattr(matched, "rng_for_panel", lambda *args, **kwargs: FixedGenerator())
    gamma = matched.generate_sample("gamma")
    lognormal = matched.generate_sample("lognormal")
    np.testing.assert_array_equal(gamma, raw-1)
    expected = (np.exp(matched.LOGNORMAL_S*z)-matched.LOGNORMAL_M)/matched.LOGNORMAL_D
    np.testing.assert_array_equal(lognormal, expected)
    assert not np.isclose(gamma.mean(), 0) and not np.isclose(gamma.var(ddof=1), 1)
    assert not np.isclose(lognormal.mean(), 0) and not np.isclose(lognormal.var(ddof=1), 1)


def test_m05_rng_namespace_population_outer_purpose_inner_exact_keys():
    assert matched.MAIN_NAMESPACE == 0x4D3035
    assert matched.SMOKE_NAMESPACE == 0x534D3035
    assert matched.DISTRIBUTION_IDS == {"gamma": 1, "lognormal": 2}
    for distribution, population_id in matched.DISTRIBUTION_IDS.items():
        actual = matched.rng_for_panel(distribution, 20, 2, 9999, 1, 0, matched.SMOKE_NAMESPACE)
        expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            entropy=2, spawn_key=(matched.SMOKE_NAMESPACE, population_id, 20, 9999, 1, 0))))
        np.testing.assert_array_equal(actual.bit_generator.random_raw(10), expected.bit_generator.random_raw(10))
    gamma_key = matched.rng_for_panel("gamma", 20, 0, 0, 0, 0, matched.SMOKE_NAMESPACE).bit_generator.random_raw(10)
    log_key = matched.rng_for_panel("lognormal", 20, 0, 0, 0, 0, matched.SMOKE_NAMESPACE).bit_generator.random_raw(10)
    boot_key = matched.rng_for_panel("gamma", 20, 0, 0, 1, 0, matched.SMOKE_NAMESPACE).bit_generator.random_raw(10)
    assert not np.array_equal(gamma_key, log_key)
    assert not np.array_equal(gamma_key, boot_key)


def test_fixed_sample_resampling_key_batch_and_order_invariance():
    first = matched.generate_bootstrap_means(SAMPLE, "lognormal", 20, 1, 7001, 0, 19,
                                             matched.SMOKE_NAMESPACE)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        entropy=1, spawn_key=(matched.SMOKE_NAMESPACE, 2, 20, 7001, 1, 0))))
    indices = rng.integers(0, 20, size=(19, 20))
    np.testing.assert_array_equal(first, SAMPLE[indices].mean(axis=1))
    matched.rng_for_panel("gamma", 20, 2, 3, 1, 0, matched.SMOKE_NAMESPACE)
    for cap in (20, 60, 1000000):
        again = matched.generate_bootstrap_means(SAMPLE, "lognormal", 20, 1, 7001, 0, 19,
                                                 matched.SMOKE_NAMESPACE, max_elements=cap)
        np.testing.assert_array_equal(first, again)


def test_single_shared_4999_distribution_matches_scipy_public_reference():
    percentile = stats.bootstrap((SAMPLE,), np.mean, n_resamples=4999, method="percentile",
                                 confidence_level=.95, rng=np.random.default_rng(74))
    bca = stats.bootstrap((SAMPLE,), np.mean, n_resamples=0, method="BCa",
                          confidence_level=.95, bootstrap_result=percentile)
    context = matched.prepare_sample(SAMPLE)
    actual = matched.infer_single(context, percentile.bootstrap_distribution)
    assert set(actual) == {"t", "percentile", "bca"}
    for method, reference in (("percentile", percentile), ("bca", bca)):
        assert actual[method]["valid"]
        assert [actual[method]["lower"], actual[method]["upper"]] == pytest.approx(
            [reference.confidence_interval.low, reference.confidence_interval.high], abs=1e-12)
    assert actual["t"] == context["t"]
    assert actual["bca"]["a"] == context["a"]


def test_fixed_percentile_linear_ties_acceleration_and_no_truth_inference_inputs():
    sample = np.array([0., 1., 2.])
    context = matched.prepare_sample(sample, alpha=.5)
    means = np.array([0., 1., 1., 2.])
    actual = matched.infer_single(context, means)
    assert [actual["percentile"]["lower"], actual["percentile"]["upper"]] == pytest.approx([.75, 1.25])
    assert actual["bca"]["z0"] == pytest.approx(0)
    assert actual["bca"]["a"] == pytest.approx(0)
    assert actual["bca"]["tail_warning"]
    assert set(inspect.signature(matched.infer_single).parameters) == {"context", "means"}
    assert set(inspect.signature(matched.prepare_sample).parameters) == {"sample", "alpha"}
    assert matched.prepare_sample is stability.prepare_sample
    assert matched.bootstrap_means is stability.bootstrap_means


@pytest.mark.parametrize("means", [None, 1., [], [[1., 2.]], [1j, 2j], ["1", "2"],
                                    [True, False], [np.nan], [np.inf], np.arange(5000)])
def test_invalid_large_or_malformed_mean_distribution_has_reasons_and_preserves_t(means):
    result = matched.infer_single(matched.prepare_sample(SAMPLE), means)
    assert result["t"]["valid"]
    for method in ("percentile", "bca"):
        assert not result[method]["valid"] and result[method]["reason"]
        assert np.isnan(result[method]["lower"]) and np.isnan(result[method]["upper"])


def test_invalid_original_remains_invalid_and_bca_never_falls_back():
    constant = matched.infer_single(matched.prepare_sample(np.full(20, .8)), np.arange(19))
    assert all(not constant[m]["valid"] and constant[m]["reason"] for m in matched.METHODS)
    bad_bca = matched.infer_single(matched.prepare_sample(SAMPLE), np.full(19, -10.))
    assert bad_bca["t"]["valid"] and bad_bca["percentile"]["valid"]
    assert not bad_bca["bca"]["valid"] and np.isnan(bad_bca["bca"]["lower"])


def one_original():
    return {"lower": np.array([-1., -3., .3]), "upper": np.array([1., -1., 2.]),
            "valid": np.ones(3, dtype=bool), "tail_warning": np.array([False, False, True])}


def test_k1_indicator_signs_partition_width_and_every_stability_field_na():
    result = matched.original_metrics(one_original())
    np.testing.assert_array_equal(result["C"], [1, 0, 0])
    np.testing.assert_array_equal(result["Mbelow"], [0, 1, 0])
    np.testing.assert_array_equal(result["Mabove"], [0, 0, 1])
    np.testing.assert_array_equal(result["T"], [0, 1, -1])
    np.testing.assert_array_equal(result["C"]+result["Mbelow"]+result["Mabove"], np.ones(3))
    np.testing.assert_allclose(result["mean_width"], [2, 2, 1.7])
    assert (result["total_inner"] == 1).all() and result["complete"].all()
    for key in ("lower_sd", "upper_sd", "S", "Ase", "Apop"):
        assert np.isnan(result[key]).all()  # Including t, with no zero fill.


def test_k1_batch_metrics_match_single_and_invalid_is_not_noncoverage():
    first = one_original()
    batch = {key: np.stack([value, value]) for key, value in first.items()}
    batch["valid"][1, 2] = False
    batch["lower"][1, 2] = batch["upper"][1, 2] = np.nan
    result = matched.original_metrics(batch)
    reference = matched.original_metrics(first)
    for key in reference:
        np.testing.assert_array_equal(result[key][0], reference[key])
    assert result["total_inner"][1, 2] == 1
    assert result["valid_inner"][1, 2] == 0 and result["invalid_inner"][1, 2] == 1
    assert not result["complete"][1, 2]
    for key in ("C", "Mbelow", "Mabove", "T", "mean_width", "warning_rate"):
        assert np.isnan(result[key][1, 2])
    assert result["complete"][1, 0] and result["complete"][1, 1]


@pytest.mark.parametrize("low,high,status,direction,equivalent", [
    (.010001, .02, "supported", "positive", False),
    (-.02, -.010001, "supported", "negative", False),
    (-.009, .009, "not_supported", "within_margin", True),
    (.001, .009, "not_supported", "within_margin", True),
    (-.009, -.001, "not_supported", "within_margin", True),
    (.01, .02, "inconclusive", "undetermined", False),
    (-.02, -.01, "inconclusive", "undetermined", False),
    (-.01, .009, "inconclusive", "undetermined", False),
    (-.009, .01, "inconclusive", "undetermined", False),
    (-.01, .01, "inconclusive", "undetermined", False),
    (-.02, .02, "inconclusive", "undetermined", False),
    (np.nan, np.nan, "inconclusive", "undetermined", False),
    (0., 0., "inconclusive", "undetermined", False),
    (.02, -.02, "inconclusive", "undetermined", False),
])
def test_strict_one_pp_difference_equivalence_and_boundary_contact(low, high, status, direction, equivalent):
    result = matched.practical_difference_status(low, high)
    assert result["status"] == status and result["direction"] == direction
    assert result["equivalence_supported"] is equivalent


def test_incomplete_or_se_zero_caller_cannot_falsely_support_equivalence():
    for bounds in ((-.009, .009), (.02, .03), (0., 0.)):
        result = matched.practical_difference_status(*bounds, complete=False)
        assert result["status"] == "inconclusive" and not result["equivalence_supported"]
    zero_se = matched.welch_interval(np.full(5, .8), np.full(5, .8))
    assert not zero_se["valid"]
    result = matched.interval_status(zero_se["low"], zero_se["high"], valid=zero_se["valid"])
    assert result["status"] == "inconclusive" and not result["equivalence_supported"]
    assert "seed" not in inspect.signature(matched.practical_difference_status).parameters


def test_original_level_welch_public_reference_and_invalid_denominator():
    gamma = np.array([0., 1., 1., 0., 1.])
    lognormal = np.array([0., 1., 1., 1., 1., 0.])
    result = matched.welch_interval(lognormal, gamma)
    reference = stats.ttest_ind(lognormal, gamma, equal_var=False)
    ci = reference.confidence_interval(confidence_level=.99)
    assert [result["low"], result["high"]] == pytest.approx([ci.low, ci.high], abs=1e-12)
    assert result["difference"] == pytest.approx(lognormal.mean()-gamma.mean())
    assert result["se"] == pytest.approx(np.sqrt(lognormal.var(ddof=1)/6+gamma.var(ddof=1)/5))
    invalid = matched.welch_interval([0., 1., np.nan], gamma)
    assert invalid["N1"] == 3 and invalid["finite_count1"] == 2
    assert not invalid["valid"] and np.isnan(invalid["difference"])


@pytest.mark.parametrize("key,value", [("n", 5), ("n", 20.), ("outer_id", -1), ("outer_id", 10000),
    ("inner_id", 1), ("purpose_id", 2), ("master_seed", -1), ("master_seed", True), ("namespace", 2**32)])
def test_invalid_rng_contract(key, value):
    with pytest.raises(ValueError):
        matched.rng_for_panel(**({"distribution": "gamma", "n": 20} | {key: value}))


@pytest.mark.parametrize("distribution", ["normal", "gamma_1", None, [], 1])
def test_unknown_population_rejected(distribution):
    with pytest.raises(ValueError):
        matched.rng_for_panel(distribution)


@pytest.mark.parametrize("margin", [0, -1, np.nan, np.inf, True, "0.01"])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError):
        matched.practical_difference_status(-.001, .001, margin=margin)


def test_resampling_bound_and_k1_shape_contract():
    for cap in (19, 1000001, True):
        with pytest.raises(ValueError):
            matched.generate_bootstrap_means(SAMPLE, "gamma", Bmax=19, max_elements=cap)
    with pytest.raises(ValueError):
        matched.generate_bootstrap_means(SAMPLE, "gamma", Bmax=5000)
    with pytest.raises(ValueError):
        matched.generate_bootstrap_means(SAMPLE[:-1], "gamma")
    with pytest.raises(ValueError):
        matched.original_metrics({key: np.zeros((2, 3, 1), dtype=value.dtype) for key, value in one_original().items()})
