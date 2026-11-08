"""Fixed-reference tests only; neither the main nor the 18-cell smoke runs here."""

import inspect
import itertools

import numpy as np
import pytest
from scipy import stats

from mathscape.bootstrap import (
    DISTRIBUTION_IDS, MAIN_NAMESPACE, MAX_ELEMENTS, METHODS, PURPOSE_IDS,
    SMOKE_NAMESPACE, bootstrap_means, draw_replicate, intervals,
)
from mathscape.inference import t_inference


SAMPLE = np.array([0., 1., 2., 3., 9.])
BOOT_MEANS = np.linspace(-1, 8, 99)


def assert_same_records(first, second):
    assert first.keys() == second.keys()
    for method in first:
        assert first[method].keys() == second[method].keys()
        for key, value in first[method].items():
            if isinstance(value, float):
                assert value == pytest.approx(second[method][key], nan_ok=True)
            else:
                assert value == second[method][key]


def test_percentile_fixed_hand_interpolation():
    # Four sorted bootstrap means and q=.25,.75 give ranks .75 and 2.25.
    result = intervals([0, 1, 2], [3, 0, 2, 1], alpha=0.5)
    assert result["percentile"]["valid"]
    assert result["percentile"]["lower"] == pytest.approx(0.75)
    assert result["percentile"]["upper"] == pytest.approx(2.25)
    assert result["percentile"]["q_low"] == 0.25
    assert result["percentile"]["q_high"] == 0.75


def test_existing_t_engine_reused_and_truth_absent_from_interval_api():
    result = intervals(SAMPLE, BOOT_MEANS)
    reference = t_inference(SAMPLE, alpha=0.05)
    assert result["t"]["valid"]
    assert result["t"]["lower"] == reference["lower"][0]
    assert result["t"]["upper"] == reference["upper"][0]
    assert not result["t"]["tail_warning"]
    assert set(inspect.signature(intervals).parameters) == {"sample", "bootstrap_distribution", "alpha"}


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("sample", [np.array([1., 2., 3., 4., 9.]), np.array([-2., -1., 0., 1., 2.]), np.array([0., 0., 1., 1., 3.])])
def test_shared_scipy_bootstrap_distribution_public_reference(sample, alpha):
    scipy_percentile = stats.bootstrap(
        (sample,), np.mean, n_resamples=99, method="percentile",
        confidence_level=1 - alpha, rng=np.random.default_rng(321),
    )
    # Reuse the exact same distribution through the public API, without new
    # draws or dependence on SciPy's private BCa helper implementation.
    scipy_bca = stats.bootstrap(
        (sample,), np.mean, n_resamples=0, method="BCa",
        confidence_level=1 - alpha, bootstrap_result=scipy_percentile,
    )
    distribution = scipy_percentile.bootstrap_distribution
    result = intervals(sample, distribution, alpha)
    assert result["percentile"]["valid"] and result["bca"]["valid"]
    for method, reference in [("percentile", scipy_percentile), ("bca", scipy_bca)]:
        actual = result[method]
        assert actual["lower"] == pytest.approx(reference.confidence_interval.low, abs=1e-12)
        assert actual["upper"] == pytest.approx(reference.confidence_interval.high, abs=1e-12)


def test_bca_acceleration_centered_sum_matches_independent_delete_one_jackknife():
    result = intervals(SAMPLE, BOOT_MEANS)["bca"]
    assert result["valid"]
    jackknife = np.array([np.mean(np.delete(SAMPLE, i)) for i in range(SAMPLE.size)])
    influence = np.mean(jackknife) - jackknife
    expected = np.sum(influence**3) / (6 * np.sum(influence**2)**1.5)
    centered = SAMPLE - np.mean(SAMPLE)
    direct = np.sum(centered**3) / (6 * np.sum(centered**2)**1.5)
    assert result["a"] == pytest.approx(expected, abs=1e-15)
    assert result["a"] == pytest.approx(direct, abs=1e-15)


def test_bias_correction_counts_half_of_exact_ties():
    sample = np.array([0., 1., 2.])  # mean=1.
    distribution = np.array([0., 0.5, 1., 1., 1.5])
    result = intervals(sample, distribution)["bca"]
    expected_probability = (2 + 0.5 * 2) / 5
    assert result["valid"]
    assert result["z0"] == pytest.approx(stats.norm.ppf(expected_probability), abs=1e-15)
    assert result["a"] == pytest.approx(0)
    assert result["tail_warning"]


def test_bca_adjusted_probabilities_match_the_defining_formula():
    result = intervals(SAMPLE, BOOT_MEANS, alpha=0.05)["bca"]
    z0, acceleration = result["z0"], result["a"]
    zs = stats.norm.ppf([0.025, 0.975])
    qs = stats.norm.cdf(z0 + (z0 + zs) / (1 - acceleration * (z0 + zs)))
    assert result["q_low"] == pytest.approx(qs[0], abs=1e-15)
    assert result["q_high"] == pytest.approx(qs[1], abs=1e-15)
    expected = np.quantile(BOOT_MEANS, qs, method="linear")
    assert [result["lower"], result["upper"]] == pytest.approx(expected, abs=1e-13)


def test_location_and_positive_scale_equivariance_for_all_three_methods():
    distribution = np.arange(-8, 33) / 4  # Exact binary fractions preserve ties.
    original = intervals(SAMPLE, distribution)
    transformed = intervals(8 + 2 * SAMPLE, 8 + 2 * distribution)
    for method in METHODS:
        assert original[method]["valid"] and transformed[method]["valid"]
        assert transformed[method]["lower"] == pytest.approx(8 + 2 * original[method]["lower"], abs=1e-12)
        assert transformed[method]["upper"] == pytest.approx(8 + 2 * original[method]["upper"], abs=1e-12)
    assert transformed["bca"]["z0"] == original["bca"]["z0"]
    assert transformed["bca"]["a"] == pytest.approx(original["bca"]["a"], abs=1e-15)


def test_finite_low_tail_resolution_is_flagged_without_clipping_or_invalidation():
    result = intervals(SAMPLE, BOOT_MEANS, alpha=0.01)
    for method in ("percentile", "bca"):
        record = result[method]
        assert record["valid"]
        assert record["tail_warning"]
        assert min(record["tail_count_low"], record["tail_count_high"]) < 10
        assert record["tail_count_low"] == pytest.approx(99 * record["q_low"])
        assert record["tail_count_high"] == pytest.approx(99 * (1 - record["q_high"]))
        ends = np.quantile(BOOT_MEANS, [record["q_low"], record["q_high"]], method="linear")
        assert [record["lower"], record["upper"]] == pytest.approx(ends)


@pytest.mark.parametrize("distribution", [[-3, -2, -1], [10, 11, 12]])
def test_infinite_bias_correction_invalidates_only_bca_without_fallback(distribution):
    result = intervals([0, 1, 2], distribution)
    assert result["t"]["valid"] and result["percentile"]["valid"]
    assert not result["bca"]["valid"]
    assert result["bca"]["reason"] == "bca_bias_correction_nonfinite"
    assert np.isnan(result["bca"]["lower"]) and np.isnan(result["bca"]["upper"])


def test_singular_adjustment_invalidates_bca_without_probability_clipping():
    sample = [0, 0, 0, 0, 1]
    distribution = np.r_[np.full(998, 0.1), 0.3]
    result = intervals(sample, distribution, alpha=1e-15)
    assert result["percentile"]["valid"]
    assert not result["bca"]["valid"]
    assert result["bca"]["reason"] == "bca_adjustment_singular"


@pytest.mark.parametrize("sample", [[], [1], [2, 2, 2], [1, np.nan, 3], [1, np.inf, 3], [[1, 2, 3]], [1j, 2j], [True, False], ["1", "2"], [1e308, 1e308, -1e308]])
def test_bad_original_sample_has_explicit_method_masks_and_reasons(sample):
    result = intervals(sample, BOOT_MEANS)
    for method in METHODS:
        assert not result[method]["valid"]
        assert result[method]["reason"]
        assert np.isnan(result[method]["lower"]) and np.isnan(result[method]["upper"])
    with pytest.raises(ValueError):
        bootstrap_means(sample, np.random.default_rng(1))


@pytest.mark.parametrize("distribution", [[], [np.nan], [np.inf], [[1, 2]], [1j, 2j], [True, False], ["1", "2"], np.arange(1000)])
def test_bad_bootstrap_distribution_preserves_t_and_invalidates_bootstrap_methods(distribution):
    result = intervals(SAMPLE, distribution)
    assert result["t"]["valid"]
    for method in ("percentile", "bca"):
        assert not result[method]["valid"]
        assert result[method]["reason"].startswith("bootstrap_distribution_")


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf, True, "0.05"])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        intervals(SAMPLE, BOOT_MEANS, alpha)


def test_alpha_and_method_display_order_do_not_mutate_shared_inputs():
    sample = SAMPLE.copy()
    distribution = BOOT_MEANS.copy()
    sample_before, distribution_before = sample.copy(), distribution.copy()
    first = intervals(sample, distribution)
    intervals(sample, distribution, alpha=0.1)
    for order in itertools.permutations(METHODS):
        again = intervals(sample, distribution)
        for method in order:
            assert_same_records({method: first[method]}, {method: again[method]})
    np.testing.assert_array_equal(sample, sample_before)
    np.testing.assert_array_equal(distribution, distribution_before)


@pytest.mark.parametrize("distribution", ["normal", "centered_exponential"])
def test_replicate_stream_is_reproducible_order_independent_and_distinct(distribution):
    first = draw_replicate(distribution, 5, 1, 7, B_boot=31, namespace=SMOKE_NAMESPACE)
    draw_replicate("normal", 20, 2, 3, B_boot=3, namespace=SMOKE_NAMESPACE)
    again = draw_replicate(distribution, 5, 1, 7, B_boot=31, namespace=SMOKE_NAMESPACE)
    for x, y in zip(first, again):
        np.testing.assert_array_equal(x, y)
    different = draw_replicate(distribution, 5, 1, 8, B_boot=31, namespace=SMOKE_NAMESPACE)
    assert not np.array_equal(first[0], different[0])
    assert not np.array_equal(first[1], different[1])
    if distribution == "centered_exponential":
        assert np.all(first[0] >= -1)


def test_exact_m02_rng_namespace_and_purpose_mapping():
    assert MAIN_NAMESPACE == 0x4D3032
    assert SMOKE_NAMESPACE == 0x534D3032
    assert DISTRIBUTION_IDS == {"normal": 1, "centered_exponential": 2}
    assert PURPOSE_IDS == {"original": 0, "bootstrap": 1}
    sample, means = draw_replicate("centered_exponential", 5, 2, 4, B_boot=9, namespace=SMOKE_NAMESPACE)
    original_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        entropy=2, spawn_key=(SMOKE_NAMESPACE, 2, 5, 4, 0))))
    bootstrap_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        entropy=2, spawn_key=(SMOKE_NAMESPACE, 2, 5, 4, 1))))
    expected_sample = original_rng.exponential(scale=1, size=5) - 1
    expected_indices = bootstrap_rng.integers(0, 5, size=(9, 5))
    np.testing.assert_array_equal(sample, expected_sample)
    np.testing.assert_array_equal(means, np.mean(sample[expected_indices], axis=1))


def test_bootstrap_batch_cap_preserves_draws_and_each_resample_has_n_elements():
    reference_rng = np.random.default_rng(73)
    indices = reference_rng.integers(0, SAMPLE.size, size=(99, SAMPLE.size))
    expected = np.mean(SAMPLE[indices], axis=1)
    for cap in (SAMPLE.size, 3 * SAMPLE.size, MAX_ELEMENTS):
        actual = bootstrap_means(SAMPLE, np.random.default_rng(73), B_boot=99, max_elements=cap)
        np.testing.assert_array_equal(actual, expected)


def test_replicate_batch_partition_and_bootstrap_count_prefix_invariance():
    def draws(ids):
        return [draw_replicate("normal", 5, 0, i, B_boot=9, namespace=SMOKE_NAMESPACE) for i in ids]
    whole = draws(range(4))
    split = draws(range(2)) + draws(range(2, 4))
    for pair1, pair2 in zip(whole, split):
        for first, second in zip(pair1, pair2):
            np.testing.assert_array_equal(first, second)
    short = draw_replicate("normal", 5, 0, 0, B_boot=3, namespace=SMOKE_NAMESPACE)
    np.testing.assert_array_equal(short[0], whole[0][0])
    np.testing.assert_array_equal(short[1], whole[0][1][:3])


@pytest.mark.parametrize("key,invalid", [
    ("n", 1), ("n", 201), ("n", 5.0), ("n", True),
    ("master_seed", -1), ("master_seed", 2**32), ("master_seed", np.nan),
    ("replicate_id", -1), ("replicate_id", 1.5), ("replicate_id", True),
    ("namespace", -1), ("namespace", 2**32), ("namespace", "main"),
    ("B_boot", 0), ("B_boot", 1000), ("B_boot", 1.5), ("B_boot", True),
])
def test_invalid_replicate_configuration_rejected(key, invalid):
    with pytest.raises(ValueError):
        draw_replicate("normal", **({"n": 5, "namespace": SMOKE_NAMESPACE} | {key: invalid}))


@pytest.mark.parametrize("distribution", ["gamma", None, [], 1])
def test_unknown_population_rejected(distribution):
    with pytest.raises(ValueError):
        draw_replicate(distribution, 5, namespace=SMOKE_NAMESPACE)


@pytest.mark.parametrize("cap", [0, 4, 1_000_001, 10.5, True])
def test_resample_array_element_limit(cap):
    with pytest.raises(ValueError):
        bootstrap_means(SAMPLE, np.random.default_rng(1), max_elements=cap)


def test_bootstrap_requires_generator():
    with pytest.raises(ValueError):
        bootstrap_means(SAMPLE, None)
