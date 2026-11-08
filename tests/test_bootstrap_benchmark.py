"""Deterministic small fixtures only; no fixed main experiment or smoke run."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import norm

from scripts.run_bootstrap_benchmark import (
    BudgetExceeded, DESIGN, MAIN_NAMESPACE, METHODS, POLICY, SMOKE_NAMESPACE,
    decide_hypotheses, enforce_budget, fixed_cells, method_summary,
    normal_t_band, pair_from_sufficient, paired_summary, pool_pairs,
    quantity_status, run_cell, validate_manifest,
)


def test_fixed_totals_and_namespaces():
    cells = fixed_cells()
    assert len(cells) == len(set(cells)) == 18
    assert len(cells) * DESIGN["R"] == 36000
    assert len(cells) * DESIGN["R"] * DESIGN["B_boot"] == 35964000
    assert sum(n * DESIGN["R"] * DESIGN["B_boot"] for _, n, _ in cells) == 1498500000
    assert MAIN_NAMESPACE == 0x4D3032
    assert SMOKE_NAMESPACE == 0x534D3032
    assert MAIN_NAMESPACE != SMOKE_NAMESPACE


def test_manifest_rejects_design_drift():
    manifest = {"schema_version": 1, "question_id": "RQ-M02", "design": deepcopy(DESIGN),
                "policy": deepcopy(POLICY), "randomness": {"main_namespace": MAIN_NAMESPACE}}
    validate_manifest(manifest)
    manifest["design"]["R"] = 2001
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_coverage_and_direction_partition_with_closed_endpoints():
    row = method_summary(np.array([-1., 0, .1, -2]), np.array([1., 1, 2, -.1]),
                         np.ones(4, bool), np.array([False, True, False, False]))
    assert row["cover_count"] == 2
    assert row["miss_above_count"] == row["miss_below_count"] == 1
    assert row["coverage"] == .5
    assert row["miss_above"] == row["miss_below"] == .25
    assert row["tail_warning_count"] == 1
    assert row["coverage_decision_low"] < row["coverage_low"]
    assert row["coverage_decision_high"] > row["coverage_high"]


def test_width_stats_match_direct_sample_calculation():
    widths = np.array([2., 3, 4, 7])
    row = method_summary(-widths/2, widths/2, np.ones(4, bool), np.zeros(4, bool))
    assert row["mean_width"] == np.mean(widths)
    assert row["width_median"] == np.median(widths)
    assert row["width_sd"] == np.std(widths, ddof=1)
    assert row["width_mcse"] == np.std(widths, ddof=1)/2
    assert row["mean_width_decision_high"] == pytest.approx(np.mean(widths)+norm.ppf(.995)*np.std(widths, ddof=1)/2)


def test_invalid_interval_preserves_total_and_na_main_estimates():
    row = method_summary(np.array([-1., np.nan]), np.array([1., np.nan]),
                         np.array([True, False]), np.array([False, False]))
    assert row["total"] == 2 and row["valid_count"] == row["invalid_count"] == 1
    assert np.isnan(row["coverage"]) and np.isnan(row["mean_width"])
    assert np.isnan(row["coverage_decision_low"])
    assert row["conditional_coverage"] == 1
    assert row["conditional_mean_width"] == 2


def _pair_data():
    return {"lower": np.array([[-1., -1, 1, 1], [-1., 1, 1, -1], [-1., -1, -1, -1]]),
            "upper": np.array([[1., 1, 3, 3], [2., 4, 4, 2], [1., 1, 1, 1]]),
            "valid": np.ones((3, 4), bool)}


def test_paired_coverage_and_width_use_corresponding_replicates():
    row = paired_summary(_pair_data(), "percentile", "t")
    differences = np.array([0., -1, 0, 1])
    assert row["coverage_diff"] == 0
    assert row["coverage_diff_mcse"] == pytest.approx(np.std(differences, ddof=1)/2)
    assert row["coverage_diff_sum"] == 0 and row["coverage_diff_sumsq"] == 2
    assert row["width_diff"] == 1 and row["width_diff_mcse"] == 0


def test_invalid_pair_is_explicitly_conditional_only():
    data = _pair_data()
    data["valid"][0, 0] = False
    row = paired_summary(data, "percentile", "t")
    assert row["total"] == 4 and row["pair_valid_count"] == 3 and row["pair_invalid_count"] == 1
    assert np.isnan(row["coverage_diff"]) and np.isnan(row["width_diff"])
    assert row["conditional_width_diff"] == 1


def test_pooling_uses_all_replicates_not_three_seed_means():
    values = [np.array([-1., 0, 1, 0]), np.array([0., 0, 0, 1]), np.array([-1., -1, 0, 1])]
    rows = [pair_from_sufficient(4, 4, x.sum(), (x*x).sum(), (2*x).sum(), ((2*x)**2).sum()) for x in values]
    pooled = pool_pairs(rows)
    all_values = np.concatenate(values)
    assert pooled["total"] == 12
    assert pooled["coverage_diff"] == pytest.approx(all_values.mean())
    assert pooled["coverage_diff_mcse"] == pytest.approx(all_values.std(ddof=1)/np.sqrt(12))
    assert pooled["width_diff_mcse"] == pytest.approx(2*all_values.std(ddof=1)/np.sqrt(12))


@pytest.mark.parametrize("low,high,threshold,direction,seeds,expected", [
    (-.05, -.03, -.02, "less", [-.03, -.04, -.05], "supported"),
    (-.05, -.03, -.02, "less", [-.01, -.04, -.05], "inconclusive"),
    (-.01, .01, -.02, "less", [-.03, .01, .02], "not_supported"),
    (-.03, -.02, -.02, "less", [-.03]*3, "inconclusive"),
    (.02, .04, .01, "greater", [.02, .03, .04], "supported"),
    (-.04, -.02, .01, "greater", [.02, .03, -.04], "not_supported"),
    (np.nan, .04, .01, "greater", [.02]*3, "inconclusive"),
])
def test_preregistered_strict_thresholds_and_seed_consistency(low, high, threshold, direction, seeds, expected):
    assert quantity_status(low, high, threshold, direction, seeds) == expected


@pytest.mark.parametrize("wall,rss,size", [(300, 1, 1), (1, 2*1024**3, 1), (1, 1, 50*1024**2)])
def test_resource_limits_stop(wall, rss, size):
    with pytest.raises(BudgetExceeded):
        enforce_budget(wall, rss, size)


def test_normal_t_prerequisite_band():
    low, high = normal_t_band()
    assert low < 1900 < high and 0 <= low < high <= 2000


def _decision_fixture():
    summaries, pairs, pooled, pooled_pairs = [], [], [], []
    for distribution, n, seed in fixed_cells():
        for method in METHODS:
            coverage = .90 if n == 5 else .94
            summaries.append(dict(distribution=distribution, n=n, master_seed=seed, method=method,
                                  invalid_count=0, cover_count=1900 if distribution == "normal" else int(2000*coverage),
                                  coverage=coverage))
        for method, reference, coverage, width in [("percentile", "t", -.05, -.5), ("bca", "percentile", .03, .1), ("bca", "t", -.02, -.4)]:
            pairs.append(dict(distribution=distribution, n=n, master_seed=seed, method=method,
                              reference=reference, coverage_diff=coverage, width_diff=width))
    for n in [5, 20, 100]:
        coverage = .90 if n == 5 else .94
        pooled.append(dict(distribution="centered_exponential", n=n, method="bca", coverage=coverage,
                           coverage_decision_low=coverage-.005, coverage_decision_high=coverage+.005))
        for method, reference, coverage, width in [("percentile", "t", -.05, -.5), ("bca", "percentile", .03, .1)]:
            pooled_pairs.append(dict(distribution="centered_exponential", n=n, method=method, reference=reference,
                                     coverage_diff=coverage, coverage_diff_decision_low=coverage-.005,
                                     coverage_diff_decision_high=coverage+.005, width_diff=width,
                                     width_diff_decision_low=width-.005, width_diff_decision_high=width+.005))
    return summaries, pairs, pooled, pooled_pairs


def test_hypotheses_need_normal_reference_and_zero_invalids():
    data = _decision_fixture()
    result = decide_hypotheses(*data)
    assert result["normal_t_reference_resolved"]
    assert all(h["final_status"] == "supported" for h in result["hypotheses"].values())
    data[0][0]["cover_count"] = 1700
    unresolved = decide_hypotheses(*data)
    assert not unresolved["normal_t_reference_resolved"]
    assert all(h["final_status"] == "inconclusive" for h in unresolved["hypotheses"].values())
    data = _decision_fixture()
    data[0][-1]["invalid_count"] = 1
    invalid = decide_hypotheses(*data)
    assert not invalid["all_cells_complete_and_valid"]
    assert all(h["final_status"] == "inconclusive" for h in invalid["hypotheses"].values())


def test_tiny_mocked_outer_loop_contract():
    calls = []
    def draw(distribution, n, **kwargs):
        calls.append((distribution, n, kwargs))
        sample = np.array([-1., -.2, .1, .4, 1.2]) + kwargs["replicate_id"]*.01
        means = np.linspace(-.8, .8, 21) + sample.mean()
        return sample, means
    data, rows, pairs = run_cell("normal", 5, 2, R=2, B_boot=21, namespace=SMOKE_NAMESPACE, draw=draw)
    assert data["lower"].shape == data["valid"].shape == (3, 2)
    assert data["reason"].dtype.kind == "U"
    assert len(rows) == len(pairs) == 3
    assert [r["method"] for r in rows] == list(METHODS)
    assert all(r["total"] == 2 for r in rows)
    assert [c[2]["replicate_id"] for c in calls] == [0, 1]
    assert all(c[2]["namespace"] == SMOKE_NAMESPACE for c in calls)


@pytest.mark.parametrize("R,B", [(0, 1), (2001, 1), (True, 1), (1, 0), (1, 1000), (1, True)])
def test_outer_loop_rejects_invalid_counts(R, B):
    with pytest.raises(ValueError):
        run_cell("normal", 5, 0, R=R, B_boot=B)
