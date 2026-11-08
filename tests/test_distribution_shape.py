"""Fixed RQ-M04 references, never the main experiment or all-cell smoke."""

import inspect

import numpy as np
import pytest
from scipy import stats

from mathscape import distribution_shape as shape
from mathscape import stability


SAMPLE = np.array([0., .1, .2, .2, .3, .5, .7, .9, 1., 1.2,
                   1.5, 1.8, 2., 2.5, 3., 3.2, 4., 5., 7., 9.])


@pytest.mark.parametrize("distribution", shape.DISTRIBUTIONS)
def test_theoretical_standardization_matches_scipy_population_moments(distribution):
    info = shape.population_info(distribution)
    reference = shape.population_distribution(distribution)
    mean, variance, skewness, excess = reference.stats(moments="mvsk")
    assert mean == pytest.approx(0, abs=1e-14)
    assert variance == pytest.approx(1, abs=1e-14)
    assert skewness == pytest.approx(info["skewness"])
    assert excess == pytest.approx(info["excess_kurtosis"])
    assert info["mean"] == 0 and info["variance"] == 1
    assert reference.support()[0] == info["lower_bound"]
    if distribution != "normal":
        r = shape.GAMMA_SHAPES[distribution]
        raw_mean, raw_variance = stats.gamma(a=r, loc=0, scale=1).stats(moments="mv")
        assert (raw_mean-r)/np.sqrt(r) == pytest.approx(0)
        assert raw_variance/r == pytest.approx(1)
        assert info["skewness"] == pytest.approx(2/np.sqrt(r))
        assert info["excess_kurtosis"] == pytest.approx(6/r)


def test_r1_is_centered_exponential_as_a_distribution_not_an_old_random_panel():
    population = shape.population_distribution("gamma_1")
    x = np.array([-2., -1., -.5, 0., 1., 4.])
    np.testing.assert_allclose(population.pdf(x), stats.expon.pdf(x, loc=-1, scale=1))
    np.testing.assert_allclose(population.cdf(x), stats.expon.cdf(x, loc=-1, scale=1))
    assert population.support()[0] == -1


def test_small_shape_support_singularity_is_preserved_and_r16_is_not_normal():
    population = shape.population_distribution("gamma_0.5")
    lower = -np.sqrt(.5)
    assert np.isposinf(population.pdf(lower))
    assert population.pdf(lower-0.01) == 0
    info = shape.population_info("gamma_16")
    assert info["skewness"] == .5 and info["excess_kurtosis"] == .375
    assert info["lower_bound"] == -4
    assert shape.population_info("normal")["lower_bound"] == -np.inf


@pytest.mark.parametrize("distribution", shape.DISTRIBUTIONS)
def test_generator_exact_mapping_and_no_per_sample_renormalization(distribution):
    actual = shape.generate_sample(distribution, 20, master_seed=2, outer_id=317,
                                   namespace=shape.SMOKE_NAMESPACE)
    sequence = np.random.SeedSequence(entropy=2,
        spawn_key=(shape.SMOKE_NAMESPACE, shape.DISTRIBUTION_IDS[distribution], 20, 317, 0, 0))
    rng = np.random.Generator(np.random.PCG64(sequence))
    if distribution == "normal":
        expected = rng.standard_normal(20)
    else:
        r = shape.GAMMA_SHAPES[distribution]
        expected = (rng.gamma(shape=r, scale=1.0, size=20)-r)/np.sqrt(r)
        assert np.all(actual >= -np.sqrt(r))
    np.testing.assert_array_equal(actual, expected)
    # Population standardization leaves the realized sample statistics variable.
    assert not np.isclose(actual.mean(), 0, atol=1e-12)
    assert not np.isclose(actual.var(ddof=1), 1, atol=1e-12)


def test_namespaces_population_ids_and_inner_purpose_separation():
    assert shape.MAIN_NAMESPACE == 0x4D3034
    assert shape.SMOKE_NAMESPACE == 0x534D3034
    assert shape.DISTRIBUTION_IDS == {"normal": 1, "gamma_0.5": 2, "gamma_1": 3,
                                      "gamma_2": 4, "gamma_4": 5, "gamma_16": 6}
    raw = [shape.rng_for_panel(d, 20, 0, 3, 1, 1, shape.SMOKE_NAMESPACE).bit_generator.random_raw(9)
           for d in shape.DISTRIBUTIONS]
    assert all(not np.array_equal(raw[a], raw[b]) for a in range(6) for b in range(a))
    original = shape.rng_for_panel("gamma_1", 20, 0, 3, 0, 0, shape.SMOKE_NAMESPACE).bit_generator.random_raw(9)
    assert not np.array_equal(original, raw[2])


def test_bootstrap_mapping_order_prefix_and_batch_invariance():
    full = shape.generate_bootstrap_means(SAMPLE, "gamma_2", 20, 1, 231, 7, 99,
                                          shape.SMOKE_NAMESPACE)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=1,
        spawn_key=(shape.SMOKE_NAMESPACE, 4, 20, 231, 1, 7))))
    indices = rng.integers(0, 20, size=(99, 20))
    np.testing.assert_array_equal(full, SAMPLE[indices].mean(axis=1))
    shape.generate_sample("normal", 20, 0, 0, shape.SMOKE_NAMESPACE)
    for cap in (20, 60, 1000000):
        short = shape.generate_bootstrap_means(SAMPLE, "gamma_2", 20, 1, 231, 7, 19,
                                               shape.SMOKE_NAMESPACE, max_elements=cap)
        np.testing.assert_array_equal(short, full[:19])
    other = shape.generate_bootstrap_means(SAMPLE, "gamma_2", 20, 1, 231, 8, 99,
                                           shape.SMOKE_NAMESPACE)
    assert not np.array_equal(full, other)


def test_old_inference_helpers_unchanged_and_default_prefixes_are_m04_only():
    assert shape.prepare_sample is stability.prepare_sample
    assert shape.bootstrap_means is stability.bootstrap_means
    assert shape.mean_t_interval is stability.mean_t_interval
    assert shape.welch_interval is stability.welch_interval
    means = np.linspace(-2, 8, 4999)
    context = shape.prepare_sample(SAMPLE)
    actual = shape.infer_prefixes(context, means)
    expected = stability.infer_prefixes(context, means, B_values=(999, 4999))
    assert set(actual) == {999, 4999}
    for b in actual:
        for method in shape.METHODS:
            for key in ("lower", "upper", "a", "z0", "q_low", "q_high", "tau_low", "tau_high"):
                assert actual[b][method][key] == pytest.approx(expected[b][method][key], nan_ok=True)
            assert actual[b][method]["valid"] == expected[b][method]["valid"]
    assert set(inspect.signature(shape.prepare_sample).parameters) == {"sample", "alpha"}


def test_shared_4999_scipy_public_distribution_matches_percentile_and_bca():
    percentile = stats.bootstrap((SAMPLE,), np.mean, n_resamples=4999, method="percentile",
                                 confidence_level=.95, rng=np.random.default_rng(31))
    bca = stats.bootstrap((SAMPLE,), np.mean, n_resamples=0, method="BCa",
                          confidence_level=.95, bootstrap_result=percentile)
    actual = shape.infer_prefixes(shape.prepare_sample(SAMPLE), percentile.bootstrap_distribution)[4999]
    for method, reference in (("percentile", percentile), ("bca", bca)):
        assert actual[method]["valid"]
        assert [actual[method]["lower"], actual[method]["upper"]] == pytest.approx(
            [reference.confidence_interval.low, reference.confidence_interval.high], abs=1e-12)


def test_t_and_acceleration_invariant_and_z0_recomputed_for_each_prefix():
    means = np.r_[np.linspace(-2, 7, 999), np.linspace(1, 10, 4000)]
    context = shape.prepare_sample(SAMPLE)
    first = shape.infer_prefixes(context, means)
    for b in first:
        assert first[b]["t"] == context["t"]
        for method in shape.METHODS:
            assert first[b][method]["a"] == context["a"]
        prefix = means[:b]
        probability = ((prefix < SAMPLE.mean()).sum()+.5*(prefix == SAMPLE.mean()).sum())/b
        assert first[b]["bca"]["z0"] == pytest.approx(stats.norm.ppf(probability))
    assert first[999]["bca"]["z0"] != first[4999]["bca"]["z0"]
    reordered = shape.infer_prefixes(context, means, (4999, 999))
    assert first[999]["bca"]["lower"] == reordered[999]["bca"]["lower"]


def endpoint_fixture(K=10):
    lower = np.full((2, 3, K), -1.)
    upper = np.full_like(lower, 1.)
    lower[:, 1, :] = np.linspace(-3, 1, K)
    upper[:, 1, :] = lower[:, 1, :] + 2
    lower[:, 2, :] = np.linspace(-4, .5, K)
    upper[:, 2, :] = lower[:, 2, :] + 2
    return {"lower": lower, "upper": upper, "valid": np.ones_like(lower, dtype=bool),
            "tail_warning": np.zeros_like(lower, dtype=bool)}


def test_signs_partition_and_population_scaled_stability_metrics():
    data = endpoint_fixture()
    se = .4
    actual = shape.original_metrics(data, se)
    assert actual["complete"].all()
    np.testing.assert_allclose(actual["C"]+actual["Mbelow"]+actual["Mabove"], 1)
    np.testing.assert_allclose(actual["T"], actual["Mbelow"]-actual["Mabove"])
    for bi in range(2):
        for mi in (1, 2):
            lo, hi = data["lower"][bi, mi], data["upper"][bi, mi]
            lower_sd, upper_sd = lo.std(ddof=1), hi.std(ddof=1)
            raw_sd = np.sqrt((lower_sd**2+upper_sd**2)/2)
            assert actual["S"][bi, mi] == pytest.approx(raw_sd)
            assert actual["Ase"][bi, mi] == pytest.approx(raw_sd/se)
            assert actual["Apop"][bi, mi] == pytest.approx(raw_sd*np.sqrt(20))
            assert actual["Mbelow"][bi, mi] == np.mean(hi < 0)
            assert actual["Mabove"][bi, mi] == np.mean(lo > 0)
    assert actual["T"][0, 2] > 0  # More intervals lie wholly below truth.


def test_population_scale_changes_evaluation_only_not_inferred_endpoints():
    data = endpoint_fixture()
    one = shape.original_metrics(data, .2)
    two = shape.original_metrics(data, .4)
    np.testing.assert_array_equal(one["S"], two["S"])
    np.testing.assert_array_equal(one["Apop"], two["Apop"])
    np.testing.assert_array_equal(one["C"], two["C"])
    np.testing.assert_allclose(one["Ase"], 2*two["Ase"])


def test_t_uses_one_original_interval_and_zero_computational_variation():
    actual = shape.original_metrics(endpoint_fixture(), .3)
    assert (actual["total_inner"][:, 0] == 1).all()
    assert (actual["total_inner"][:, 1:] == 10).all()
    for key in ("lower_sd", "upper_sd", "S", "Ase", "Apop"):
        assert (actual[key][:, 0] == 0).all()


def test_invalid_inner_keeps_denominator_and_na_metrics_only_for_affected_method_b():
    data = endpoint_fixture()
    data["valid"][0, 2, 3] = False
    actual = shape.original_metrics(data, .3)
    assert not actual["complete"][0, 2]
    assert actual["valid_inner"][0, 2] == 9 and actual["invalid_inner"][0, 2] == 1
    for key in ("C", "Mbelow", "Mabove", "T", "S", "Ase", "Apop", "mean_width", "warning_rate"):
        assert np.isnan(actual[key][0, 2])
    assert actual["complete"][1, 2] and actual["complete"][0, 1]


def test_original_unit_mcse_and_independent_shape_welch_not_paired_outer_ids():
    coverage = np.array([.2, .5, .8, 1.])
    summary = shape.mean_t_interval(coverage)
    assert summary["N"] == 4
    assert summary["mcse"] == pytest.approx(coverage.std(ddof=1)/np.sqrt(4))
    # Identical outer IDs do not justify a paired SD, even if this fixture's
    # two arrays happen to differ by a constant.
    one, zero = coverage, coverage+.1
    actual = shape.welch_interval(one, zero)
    assert actual["valid"]
    assert actual["se"] == pytest.approx(np.sqrt(one.var(ddof=1)/4+zero.var(ddof=1)/4))
    assert actual["se"] > (one-zero).std(ddof=1)/np.sqrt(4)
    reference = stats.ttest_ind(one, zero, equal_var=False).confidence_interval(confidence_level=.99)
    assert [actual["low"], actual["high"]] == pytest.approx([reference.low, reference.high])


@pytest.mark.parametrize("low,high,threshold,direction,seeds,expected", [
    (-.08, -.025, -.02, "less", [-.01, -.005, -.03], "supported"),
    (.04, .08, .03, "greater", [.01, .005, .05], "supported"),
    (-.08, -.025, -.02, "less", [-.01, 0, -.03], "inconclusive"),
    (.04, .08, .03, "greater", [.01, 0, .05], "inconclusive"),
    (-.08, -.02, -.02, "less", [-.01, -.02, -.03], "inconclusive"),
    (.03, .08, .03, "greater", [.01, .02, .03], "inconclusive"),
    (-.01, .01, -.02, "less", [-.01, -.02, -.03], "not_supported"),
    (-.01, .02, .03, "greater", [.01, .02, .03], "not_supported"),
    (np.nan, np.nan, .03, "greater", [.01, .02, .03], "inconclusive"),
])
def test_confirmatory_pooled_threshold_differs_from_zero_seed_direction(low, high, threshold, direction, seeds, expected):
    assert shape.quantity_status(low, high, threshold, direction, seeds) == expected


def test_explicit_seed_threshold_and_strict_invalid_mean_helper():
    assert shape.quantity_status(-.08, -.025, -.02, "less", [-.01, -.005, -.03], seed_threshold=-.02) == "inconclusive"
    invalid = shape.mean_t_interval([.1, np.nan, .3])
    assert invalid["N"] == 3 and invalid["finite_count"] == 2 and not invalid["valid"]
    assert np.isnan(invalid["mean"])
    constant = shape.mean_t_interval(np.full(1200, .8))
    assert not constant["valid"] and constant["sd"] == 0


@pytest.mark.parametrize("key,value", [("n", 5), ("n", 20.), ("outer_id", -1), ("outer_id", 400),
    ("inner_id", 10), ("purpose_id", 2), ("master_seed", -1), ("master_seed", True),
    ("namespace", 2**32), ("namespace", np.nan)])
def test_invalid_rng_arguments(key, value):
    with pytest.raises(ValueError):
        shape.rng_for_panel(**({"distribution": "normal", "n": 20} | {key: value}))


@pytest.mark.parametrize("distribution", ["gamma", "gamma_3", "centered_exponential", None, [], 1])
def test_unknown_population_rejected(distribution):
    with pytest.raises(ValueError):
        shape.generate_sample(distribution, namespace=shape.SMOKE_NAMESPACE)


def test_extended_outer_ids_and_existing_generators_are_unchanged():
    assert shape.generate_sample("normal", 20, outer_id=399, namespace=shape.SMOKE_NAMESPACE).shape == (20,)
    with pytest.raises(ValueError):
        stability.rng_for_panel("normal", 20, outer_id=100)
    with pytest.raises(ValueError):
        shape.rng_for_panel("normal", 20, purpose_id=0, inner_id=1)
    with pytest.raises(ValueError):
        shape.original_metrics(endpoint_fixture(), .3, n=5)
    with pytest.raises(ValueError):
        shape.generate_bootstrap_means([1, 2, 3], "normal", 20)
    for cap in (19, 1000001, True):
        with pytest.raises(ValueError):
            shape.generate_bootstrap_means(SAMPLE, "normal", 20, Bmax=19, max_elements=cap)
