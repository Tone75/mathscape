"""Deterministic references; the preregistered stochastic study runs separately."""

import inspect
import struct

import numpy as np
import pytest
from scipy import stats

from mathscape.inference import (
    BOUNDARY_ATOL, BOUNDARY_RTOL, DISTRIBUTION_IDS, generate_samples,
    normal_power, rng_for_cell, sample_batch, summarize, t_inference, wilson_interval,
)


def test_hand_calculated_sample_uses_sample_variance_and_unknown_sigma():
    # mean=3; squared deviations sum=10; s²=10/4; SE=sqrt(1/2).
    result = t_inference([1, 2, 3, 4, 5], mu0=2)
    assert result["mean"] == pytest.approx([3])
    assert result["variance"] == pytest.approx([2.5])
    assert result["sd"] == pytest.approx([np.sqrt(2.5)])
    assert result["se"] == pytest.approx([np.sqrt(0.5)])
    assert result["t"] == pytest.approx([np.sqrt(2)])
    # Tabulated t_(4,.975), not the normal 1.96 quantile.
    half_width = 2.7764451051977987 * np.sqrt(0.5)
    assert result["lower"] == pytest.approx([3 - half_width], rel=1e-11)
    assert result["upper"] == pytest.approx([3 + half_width], rel=1e-11)
    assert result["valid"].tolist() == [True]
    assert result["reject"].tolist() == [False]
    assert set(inspect.signature(t_inference).parameters) == {"samples", "mu0", "alpha"}


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_independent_scipy_ttest_reference_including_confidence_interval(alpha):
    samples = np.array([
        [2.1, -0.3, 0.7, 4.0, 1.9],
        [-1.2, -2.5, -0.1, -3.1, -2.0],
        [10.0, 11.0, 9.5, 12.0, 10.8],
    ])
    result = t_inference(samples, mu0=0.2, alpha=alpha)
    reference = stats.ttest_1samp(samples, popmean=0.2, axis=1)
    interval = reference.confidence_interval(confidence_level=1 - alpha)
    np.testing.assert_allclose(result["t"], reference.statistic, rtol=1e-13)
    np.testing.assert_allclose(result["p"], reference.pvalue, rtol=1e-13)
    np.testing.assert_allclose(result["lower"], interval.low, rtol=1e-12)
    np.testing.assert_allclose(result["upper"], interval.high, rtol=1e-12)


def test_one_dimensional_input_yields_one_row_arrays():
    result = t_inference([0, 1, 2])
    for array in result.values():
        assert array.shape == (1,)
    assert result["valid"].dtype == result["reject"].dtype == np.dtype(bool)


@pytest.mark.parametrize("distribution", ["normal", "centered_exponential"])
def test_seed_reproducibility_cell_order_and_trial_ids(distribution):
    first = generate_samples(distribution, 20, 30, d=0.2, seed=7)
    generate_samples("normal", 5, 10, d=0.5, seed=1)
    again = generate_samples(distribution, 20, 30, d=0.2, seed=7)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, generate_samples(distribution, 20, 30, d=0.2, seed=8))
    assert not np.array_equal(first, generate_samples(distribution, 20, 30, d=0.2, seed=7, trial_id=1))
    assert not np.array_equal(first, generate_samples(distribution, 20, 30, d=0.5, seed=7))


@pytest.mark.parametrize("distribution", ["normal", "centered_exponential"])
def test_batch_partition_and_prefix_invariance(distribution):
    expected = generate_samples(distribution, 20, 70, d=0.2, seed=2)
    rng = rng_for_cell(distribution, 20, 0.2, 2)
    batches = [sample_batch(rng, distribution, 20, size, d=0.2) for size in (7, 31, 32)]
    np.testing.assert_array_equal(np.vstack(batches), expected)
    np.testing.assert_array_equal(generate_samples(distribution, 20, 7, d=0.2, seed=2), expected[:7])


def test_explicit_seedsequence_mapping_and_signed_zero_canonicalization():
    assert DISTRIBUTION_IDS == {"normal": 1, "centered_exponential": 2}
    bits = struct.unpack("<Q", struct.pack("<d", 0.2))[0]
    sequence = np.random.SeedSequence(7, spawn_key=(2, 20, bits & 0xFFFFFFFF, bits >> 32, 4))
    reference = np.random.Generator(np.random.PCG64(sequence))
    actual = rng_for_cell("centered_exponential", 20, 0.2, 7, trial_id=4)
    np.testing.assert_array_equal(actual.bit_generator.random_raw(20), reference.bit_generator.random_raw(20))
    np.testing.assert_array_equal(
        rng_for_cell("normal", 5, -0.0, 0).bit_generator.random_raw(10),
        rng_for_cell("normal", 5, 0.0, 0).bit_generator.random_raw(10),
    )


@pytest.mark.parametrize("distribution", ["normal", "centered_exponential"])
def test_population_transform_has_the_specified_mean_and_variance(distribution):
    # Exact transform of the same standardized draws, not a stochastic moment test.
    rng = np.random.Generator(np.random.PCG64(13))
    reference_rng = np.random.Generator(np.random.PCG64(13))
    z = (reference_rng.standard_normal((3, 5)) if distribution == "normal"
         else reference_rng.exponential(1, (3, 5)) - 1)
    actual = sample_batch(rng, distribution, 5, 3, d=0.2, mu0=2, sigma=3)
    np.testing.assert_array_equal(actual, 2 + 3 * (0.2 + z))
    if distribution == "centered_exponential":
        assert (actual >= 2 + 3 * (0.2 - 1)).all()


def test_alpha_change_preserves_samples_t_and_p_and_changes_only_inference_thresholds():
    samples = generate_samples("normal", 20, 40, d=0.2)
    before = samples.copy()
    conservative = t_inference(samples, alpha=0.01)
    liberal = t_inference(samples, alpha=0.10)
    np.testing.assert_array_equal(samples, before)
    for key in ("mean", "variance", "sd", "se", "t", "p"):
        np.testing.assert_array_equal(conservative[key], liberal[key])
    assert np.all(conservative["lower"] < liberal["lower"])
    assert np.all(conservative["upper"] > liberal["upper"])
    assert np.all(~conservative["reject"] | liberal["reject"])


def test_ci_and_test_decisions_agree_away_from_documented_rounding_boundary():
    alpha = 0.05
    critical = stats.t.isf(alpha / 2, 2)
    # Last row is constructed exactly on the mathematical rejection boundary.
    base = np.array([-1., 0., 1.])
    samples = np.vstack([base, base + 1, base + 10, base - 10, base + critical / np.sqrt(3)])
    result = t_inference(samples, alpha=alpha)
    boundary = np.isclose(np.abs(result["t"]), critical, atol=BOUNDARY_ATOL, rtol=BOUNDARY_RTOL)
    outside = (result["lower"] > 0) | (result["upper"] < 0)
    assert boundary.tolist() == [False, False, False, False, True]
    np.testing.assert_array_equal(result["reject"][~boundary], outside[~boundary])
    np.testing.assert_array_equal(result["reject"], result["p"] < alpha)
    assert abs(result["lower"][-1]) < 1e-12


def test_invalid_rows_remain_undefined_and_keep_original_denominator():
    samples = np.array([[1, 2, 3], [2, 2, 2], [0, np.nan, 2], [1, np.inf, 3]])
    result = t_inference(samples)
    assert result["valid"].tolist() == [True, False, False, False]
    assert result["variance"][1] == 0
    for key in ("t", "p", "lower", "upper"):
        assert np.isnan(result[key][1:]).all()
    assert not result["reject"][1:].any()
    summary = summarize(result, true_mean=2)
    assert summary["denominator"] == 4
    assert summary["invalid_count"] == 3
    assert summary["cover_count"] == 1
    assert summary["reject_count"] == 0
    for key in ("reject_rate", "coverage", "mean_width", "width_sd", "width_mcse", "reject_low", "coverage_high"):
        assert np.isnan(summary[key])


def test_overflowed_sample_intermediates_are_invalid_not_fake_significance():
    result = t_inference([[1e308, -1e308], [1e200, 1e200 + 1e190]])
    assert not result["valid"].any()
    assert np.isnan(result["p"]).all()


def test_summary_covers_true_mean_and_uses_all_rows_and_ddof_one_width_sd():
    # At true_mean=2 only the third interval covers; at the null=0 two cover.
    result = {
        "valid": np.array([True, True, True]), "reject": np.array([False, False, True]),
        "lower": np.array([-1., -0.5, 1.]), "upper": np.array([1., 0.5, 3.]),
    }
    summary = summarize(result, true_mean=2)
    assert summary["denominator"] == 3
    assert summary["invalid_count"] == 0
    assert summary["reject_count"] == summary["cover_count"] == 1
    assert summary["reject_rate"] == summary["coverage"] == pytest.approx(1 / 3)
    assert summary["mean_width"] == pytest.approx(5 / 3)
    assert summary["width_sd"] == pytest.approx(np.sqrt(1 / 3))
    assert summary["width_mcse"] == pytest.approx(1 / 3)
    assert summary["reject_mcse"] == summary["coverage_mcse"] == pytest.approx(np.sqrt(2 / 27))
    expected = wilson_interval(1, 3)
    assert (summary["reject_low"], summary["reject_high"]) == pytest.approx(expected)
    assert (summary["coverage_low"], summary["coverage_high"]) == pytest.approx(expected)


def test_single_repetition_retains_rate_but_width_sd_mcse_undefined():
    summary = summarize(t_inference([1, 2, 3]), true_mean=2)
    assert summary["denominator"] == 1
    assert summary["coverage"] == 1
    assert np.isnan(summary["width_sd"])
    assert np.isnan(summary["width_mcse"])


@pytest.mark.parametrize("count,total", [(0, 100), (100, 100), (7, 100), (1, 1), (0, 1)])
def test_wilson_matches_independent_binomtest_interval_including_zero_all(count, total):
    actual = wilson_interval(count, total)
    expected = stats.binomtest(count, total).proportion_ci(confidence_level=0.95, method="wilson")
    assert actual == pytest.approx((expected.low, expected.high), abs=1e-14)
    assert 0 <= actual[0] <= actual[1] <= 1
    assert actual[1] > actual[0]


@pytest.mark.parametrize("n", [5, 20, 100])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_normal_null_theoretical_rejection_is_alpha(n, alpha):
    assert normal_power(n, 0, alpha) == pytest.approx(alpha, abs=1e-12)


def test_normal_power_increases_with_n_for_nonzero_fixed_effect_and_is_symmetric():
    powers = [normal_power(n, 0.5) for n in (5, 20, 100)]
    assert 0.05 < powers[0] < powers[1] < powers[2] < 1
    assert normal_power(20, -0.5) == pytest.approx(normal_power(20, 0.5), abs=1e-14)


@pytest.mark.parametrize("invalid", [-1, 2**32, 1.5, True, np.nan, np.inf, "0"])
@pytest.mark.parametrize("key", ["seed", "trial_id"])
def test_invalid_seed_and_trial_id_rejected(key, invalid):
    with pytest.raises(ValueError):
        generate_samples("normal", 20, **{key: invalid})


@pytest.mark.parametrize("invalid", [0, 1, 201, 2.5, True, np.nan, np.inf, "20"])
def test_invalid_n_rejected(invalid):
    for call in (lambda: generate_samples("normal", invalid), lambda: normal_power(invalid, 0.2)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("invalid", [0, -1, 5001, 1.5, True, np.nan, np.inf, "500"])
def test_invalid_repetition_count_rejected(invalid):
    with pytest.raises(ValueError):
        generate_samples("normal", 20, repetitions=invalid)


@pytest.mark.parametrize("invalid", [0, 1, -0.1, 1.1, np.nan, np.inf, True, "0.05"])
def test_invalid_alpha_rejected(invalid):
    for call in (lambda: t_inference([1, 2, 3], alpha=invalid), lambda: normal_power(20, 0.2, alpha=invalid)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("invalid", [-1.1, 1.1, np.nan, np.inf, True, "0.2"])
def test_invalid_effect_rejected(invalid):
    for call in (lambda: generate_samples("normal", 20, d=invalid), lambda: normal_power(20, invalid)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf, True, "1"])
def test_invalid_sigma_rejected(invalid):
    with pytest.raises(ValueError):
        generate_samples("normal", 20, sigma=invalid)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, True, "0"])
def test_invalid_null_mean_rejected(invalid):
    for call in (lambda: generate_samples("normal", 20, mu0=invalid), lambda: t_inference([1, 2, 3], mu0=invalid)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("samples", [[], [1], [[1]], np.zeros((2, 201)), np.zeros((5001, 2)), [[1, 2], [1]], [True, False], [1 + 1j, 2], ["1", "2"], np.zeros((2, 2, 2))])
def test_invalid_sample_shapes_or_types_rejected(samples):
    with pytest.raises(ValueError):
        t_inference(samples)


@pytest.mark.parametrize("distribution", ["gamma", "unknown", None, [], 1])
def test_unknown_distribution_rejected(distribution):
    with pytest.raises(ValueError):
        generate_samples(distribution, 20)


@pytest.mark.parametrize("kwargs", [{"count": -1, "total": 10}, {"count": 11, "total": 10}, {"count": 1, "total": 0}, {"count": 1.5, "total": 10}, {"count": True, "total": 10}, {"count": 1, "total": 10, "confidence": 0}, {"count": 1, "total": 10, "confidence": 1}, {"count": 1, "total": 10, "confidence": np.nan}])
def test_wilson_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        wilson_interval(**kwargs)


def test_summary_invalid_inputs():
    with pytest.raises(ValueError):
        summarize({}, 0)
    with pytest.raises(ValueError):
        summarize(t_inference([1, 2, 3]), np.nan)
    malformed = {"valid": np.array([True]), "reject": np.array([False]), "lower": np.array([2.]), "upper": np.array([1.])}
    with pytest.raises(ValueError):
        summarize(malformed, 0)


def test_maximum_ui_shape_is_allowed_without_running_the_fixed_experiment():
    samples = generate_samples("normal", 200, repetitions=5000)
    assert samples.shape == (5000, 200)
    assert samples.size == 1_000_000


def test_sample_batch_requires_generator_and_overflow_is_explicit():
    with pytest.raises(ValueError):
        sample_batch(None, "normal", 20, 10)
    with pytest.raises(ValueError, match="float64"):
        sample_batch(np.random.default_rng(0), "normal", 20, 10, mu0=1e308, sigma=1e308)
