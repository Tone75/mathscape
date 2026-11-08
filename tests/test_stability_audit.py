"""Deterministic independent-audit checks; no experiment or resampling runs."""

import numpy as np
import pytest
from scipy import stats
import json

from scripts import audit_stability as audit


def fixed_panel(r=3, k=20):
    samples = np.array([np.arange(-2, 3, dtype=float) * (1 + i / 10) + i / 5 for i in range(r)])
    shape = (r, 3, 3, k)
    data = {name: np.full(shape, np.nan) for name in
            ("lower", "upper", "a", "z0", "q_low", "q_high", "tau_low", "tau_high")}
    data.update(samples=samples, sample_sha256=np.array([audit.sample_hash(sample) for sample in samples]),
                outer_id=np.arange(r), inner_id=np.arange(k), B_values=np.array(audit.BS),
                methods=np.array(audit.METHODS), distribution=np.array("centered_exponential"), n=np.array(5),
                master_seed=np.array(0), R=np.array(r), K=np.array(k), alpha=np.array(0.05),
                namespace=np.array(audit.MAIN_NAMESPACE), valid=np.ones(shape, dtype=bool),
                reason=np.full(shape, "", dtype="U40"), tail_warning=np.zeros(shape, dtype=bool))
    for i, sample in enumerate(samples):
        se = np.std(sample, ddof=1) / np.sqrt(5)
        critical = stats.t.ppf(0.975, 4)
        data["lower"][i, :, 0] = np.mean(sample) - critical * se
        data["upper"][i, :, 0] = np.mean(sample) + critical * se
        for bi, b in enumerate(audit.BS):
            for mi in (1, 2):
                data["lower"][i, bi, mi] = -0.8 + np.arange(k) / 100 + i / 10
                data["upper"][i, bi, mi] = data["lower"][i, bi, mi] + 1.6
                data["q_low"][i, bi, mi] = 0.025
                data["q_high"][i, bi, mi] = 0.975
                data["tau_low"][i, bi, mi] = b * 0.025
                data["tau_high"][i, bi, mi] = b * 0.025
                data["tail_warning"][i, bi, mi] = b * 0.025 < 10
                if mi == 2:
                    data["a"][i, bi, mi] = 0
                    data["z0"][i, bi, mi] = 0
    return data


def test_panel_audit_reconstructs_hash_se_t_acceleration_quantiles_and_tau():
    panel = fixed_panel()
    panel["se"] = np.std(panel["samples"], axis=1, ddof=1) / np.sqrt(5)
    assert audit.inspect_panel(panel, strict_design=False) == []
    assert "main R/n/K differs from fixed design" in audit.inspect_panel(panel)


@pytest.mark.parametrize("field, mutate, expected", [
    ("sample_sha256", lambda array: array.__setitem__(0, "0" * 64), "hash mismatch"),
    ("lower", lambda array: array.__setitem__((0, 1, 0, 0), -99), "t reference mismatch"),
    ("a", lambda array: array.__setitem__((0, 1, 2, 0), 0.1), "acceleration differs"),
    ("z0", lambda array: array.__setitem__((0, 1, 2, 0), 0.1234), "tie-count lattice"),
    ("q_high", lambda array: array.__setitem__((0, 1, 2, 0), 0.025), "quantile range"),
    ("tau_low", lambda array: array.__setitem__((0, 1, 2, 0), 99), "lower tau differs"),
    ("tail_warning", lambda array: array.__setitem__((0, 0, 2, 0), False), "warning mask differs"),
    ("inner_id", lambda array: array.__setitem__(0, 19), "inner IDs"),
])
def test_panel_audit_detects_corrupted_saved_contracts(field, mutate, expected):
    panel = fixed_panel()
    mutate(panel[field])
    assert any(expected in error for error in audit.inspect_panel(panel, strict_design=False))


def test_original_metrics_have_hand_calculated_sd_A_C_and_miss_sides():
    panel = fixed_panel(r=1, k=4)
    panel["lower"][0, 0, 1] = [-2, -1, 0, 1]
    panel["upper"][0, 0, 1] = [0, 1, 2, 3]
    row = next(row for row in audit.original_metrics(panel) if row["B"] == 199 and row["method"] == "percentile")
    assert row["lower_sd"] == pytest.approx(np.sqrt(5 / 3))
    assert row["upper_sd"] == pytest.approx(np.sqrt(5 / 3))
    assert row["se"] == pytest.approx(np.sqrt(0.5))
    assert row["A"] == pytest.approx(np.sqrt(10 / 3))
    assert row["C"] == 0.75
    assert row["miss_above"] == 0.25
    assert row["miss_below"] == 0
    assert row["mean_width"] == 2
    assert row["C"] + row["miss_above"] + row["miss_below"] == 1


def test_t_is_counted_once_per_original_and_has_zero_conditional_sd():
    panel = fixed_panel(r=3, k=20)
    rows = [row for row in audit.original_metrics(panel) if row["B"] == 199 and row["method"] == "t"]
    assert all(row["total_inner"] == row["valid_inner"] == 1 for row in rows)
    assert all(row["A"] == row["lower_sd"] == row["upper_sd"] == 0 for row in rows)
    summary = audit.summarize_originals(rows)
    assert summary["original_total"] == summary["interval_count"] == summary["valid_count"] == 3
    assert summary["A"] == summary["A_sd"] == summary["A_mcse"] == 0
    assert summary["A_decision_low"] is None  # constant variance does not define a confirmation CI


def test_mcse_uses_original_level_cluster_means_not_RK_intervals():
    base = audit.original_metrics(fixed_panel(r=3, k=20))
    rows = [row for row in base if row["B"] == 199 and row["method"] == "bca"]
    for row, coverage in zip(rows, [0, 0.5, 1]):
        row["C"] = coverage
    summary = audit.summarize_originals(rows)
    assert summary["coverage"] == 0.5
    assert summary["coverage_mcse"] == pytest.approx(0.5 / np.sqrt(3))
    assert summary["coverage_mcse"] != pytest.approx(np.sqrt(0.5 * 0.5 / 60))


def test_invalid_inner_retained_and_primary_NA_conditional_denominator_explicit():
    panel = fixed_panel(r=3)
    panel["valid"][0, 0, 2, 1] = False
    panel["reason"][0, 0, 2, 1] = "undefined adjusted quantile"
    panel["lower"][0, 0, 2, 1] = panel["upper"][0, 0, 2, 1] = np.nan
    rows = [row for row in audit.original_metrics(panel) if row["B"] == 199 and row["method"] == "bca"]
    assert not rows[0]["complete"] and rows[0]["A"] is None and rows[0]["C"] is None
    summary = audit.summarize_originals(rows)
    assert summary["original_total"] == 3
    assert summary["complete_original_count"] == summary["conditional_original_count"] == 2
    assert summary["interval_count"] == 60 and summary["invalid_count"] == 1
    assert summary["coverage"] is None and summary["A"] is None
    assert summary["conditional_coverage"] is not None


def test_H2_classification_and_evaluation_use_separate_k_halves():
    panel = fixed_panel(r=3, k=20)
    panel["tail_warning"][:, 1, 2] = False
    panel["tail_warning"][0, 1, 2, :5] = True
    panel["tail_warning"][1, 1, 2, :4] = True
    before = audit.h2_split(panel)
    assert [row["W"] for row in before] == [1, 0, 0]
    panel["lower"][:, 1, 2, :10] += np.arange(10) * 99
    after = audit.h2_split(panel)
    assert [row["A_eval"] for row in after] == [row["A_eval"] for row in before]
    panel["tail_warning"][:, 1, 2, 10:] = True
    assert [row["W"] for row in audit.h2_split(panel)] == [row["W"] for row in before]


def test_H2_invalid_classification_does_not_erase_independent_valid_evaluation():
    panel = fixed_panel()
    panel["valid"][0, 1, 2, 0] = False
    panel["valid"][1, 1, 2, 19] = False
    rows = audit.h2_split(panel)
    assert rows[0]["W"] is None and rows[0]["A_eval"] is not None and not rows[0]["complete"]
    assert rows[1]["W"] is not None and rows[1]["A_eval"] is None and not rows[1]["complete"]


def test_mean_t_interval_unclipped_and_zero_variance_undefined():
    reference = audit.mean_interval([0, 0.5, 1])
    critical = stats.t.ppf(0.995, 2)
    assert reference["low"] == pytest.approx(0.5 - critical * 0.5 / np.sqrt(3))
    assert reference["low"] < 0 and reference["high"] > 1
    # A mathematically constant fixture should be treated as zero variance,
    # without relying on cancellation from a non-representable decimal.
    exact_constant = audit.mean_interval([0.5] * 300)
    assert exact_constant["mean"] == 0.5 and not exact_constant["defined"]
    assert not audit.mean_interval([0, np.nan, 1])["defined"]


def test_welch_interval_matches_hand_computed_independent_group_formula():
    first, second = np.array([1, 2, 3, 4]), np.array([0, 1, 2])
    result = audit.welch_interval(first, second)
    v1, v0 = np.var(first, ddof=1), np.var(second, ddof=1)
    variance = v1 / 4 + v0 / 3
    df = variance**2 / ((v1 / 4) ** 2 / 3 + (v0 / 3) ** 2 / 2)
    half = stats.t.ppf(0.995, df) * np.sqrt(variance)
    assert result["difference"] == pytest.approx(1.5)
    assert result["df"] == pytest.approx(df)
    assert result["low"] == pytest.approx(1.5 - half)
    assert result["high"] == pytest.approx(1.5 + half)


@pytest.mark.parametrize("interval,threshold,direction,seeds,complete,groups,expected", [
    (dict(low=-0.2, high=-0.1, defined=True), 0, "less", [-0.1] * 3, True, True, "supported"),
    (dict(low=0.1, high=0.2, defined=True), 0, "less", [None] * 3, True, True, "not_supported"),
    (dict(low=-0.1, high=0.1, defined=True), 0, "less", [-0.1] * 3, True, True, "inconclusive"),
    (dict(low=0.03, high=0.08, defined=True), 0.02, "greater", [0.03] * 3, True, True, "supported"),
    (dict(low=0.03, high=0.08, defined=True), 0.02, "greater", [0.03, 0.01, 0.03], True, True, "inconclusive"),
    (dict(low=-0.1, high=0.01, defined=True), 0.02, "greater", [0.0] * 3, True, False, "inconclusive"),
    (dict(low=0.1, high=0.2, defined=True), 0, "less", [0.1] * 3, False, True, "inconclusive"),
])
def test_decisions_keep_group_and_invalid_gates_and_clear_opposite_evidence(
    interval, threshold, direction, seeds, complete, groups, expected
):
    assert audit.directional_decision(interval, threshold, direction, seeds, complete=complete, groups_ok=groups) == expected


def test_saved_table_comparison_detects_values_NA_missing_and_duplicate_keys():
    expected = [dict(id=1, value=0.5, count=20, undefined=None, complete=True)]
    observed = [dict(id="1", value="0.5", count="20", undefined="nan", complete="True")]
    assert audit.compare_rows(expected, observed, ["id"]) == []
    observed[0]["count"] = "19"
    assert any("count differs" in error for error in audit.compare_rows(expected, observed, ["id"]))
    observed[0]["undefined"] = "0"
    assert any("undefined differs" in error for error in audit.compare_rows(expected, observed, ["id"]))
    assert any("duplicate" in error for error in audit.compare_rows(expected, observed * 2, ["id"]))
    assert any("missing/extra keys" in error for error in audit.compare_rows(expected, [], ["id"]))


def confirmation_fixture():
    originals, h2_rows = [], []
    for seed in (0, 1, 2):
        for outer in range(100):
            for n in (5, 20):
                for b in (199, 4999):
                    originals.append(dict(distribution="centered_exponential", n=n, master_seed=seed,
                                          outer_id=outer, B=b, method="bca", complete=True,
                                          A=0.8 + outer * 0.001 if b == 199 else 0.2 + outer * 0.0001,
                                          C=0.8 + (outer % 3) * 0.02))
            h2_rows.append(dict(master_seed=seed, outer_id=outer, W=int(outer < 50),
                                A_eval=0.2 + outer * 0.001 + (0.1 if outer < 50 else 0), complete=True))
    return originals, h2_rows


def test_four_confirmatory_quantities_have_original_N300_and_correct_seed_gates():
    originals, h2_rows = confirmation_fixture()
    quantities = audit.confirmation_quantities(originals, h2_rows)
    assert all(quantities[name]["status"] == "supported" for name in ("H1_n5", "H1_n20", "H1", "H2", "H3"))
    assert quantities["H1_n5"]["interval"]["N"] == 300
    assert quantities["H3"]["interval"]["N"] == 300
    assert quantities["H2"]["interval"]["N1"] == quantities["H2"]["interval"]["N0"] == 150
    assert quantities["H3"]["interval"]["mean"] == pytest.approx(np.mean([row["C"] for row in originals if row["n"] == 5 and row["B"] == 4999]))


def test_H1_clear_counterexample_preserved_alongside_other_n_missing():
    originals, h2_rows = confirmation_fixture()
    for row in originals:
        if row["n"] == 5 and row["B"] == 4999:
            row["A"] = 2 + row["outer_id"] * 0.001
        if row["n"] == 20 and row["outer_id"] == 0:
            row["complete"] = False
            row["A"] = None
    quantities = audit.confirmation_quantities(originals, h2_rows)
    assert quantities["H1_n5"]["status"] == "not_supported"
    assert quantities["H1_n20"]["status"] == "inconclusive"
    assert quantities["H1"]["status"] == "not_supported"
    assert not quantities["H1_n20"]["complete"]


def test_H2_seed_group_shortage_is_inconclusive_even_with_otherwise_clear_difference():
    originals, h2_rows = confirmation_fixture()
    for row in h2_rows:
        if row["master_seed"] == 0:
            row["W"] = int(row["outer_id"] < 4)
    quantity = audit.confirmation_quantities(originals, h2_rows)["H2"]
    assert not quantity["groups_ok"]
    assert quantity["status"] == "inconclusive"


def test_H3_missing_inner_result_is_primary_NA_never_noncoverage():
    originals, h2_rows = confirmation_fixture()
    for row in originals:
        if row["n"] == 5 and row["master_seed"] == row["outer_id"] == 0 and row["B"] == 4999:
            row["complete"], row["C"] = False, None
    quantity = audit.confirmation_quantities(originals, h2_rows)["H3"]
    assert quantity["status"] == "inconclusive"
    assert quantity["interval"]["mean"] is None
    assert quantity["interval"]["N"] == 300


def test_H2_group_summary_missing_original_keeps_expected_denominator_and_primary_NA():
    panel = fixed_panel(r=3)
    cohort = audit.h2_split(panel)
    rows = audit.group_summary(cohort)
    seed = next(row for row in rows if row["master_seed"] == 0 and row["W"] == 0)
    pooled = next(row for row in rows if row["master_seed"] == "pooled" and row["W"] == 0)
    assert seed["original_total"] == 100 and seed["observed_original_count"] == 3
    assert seed["missing_original_count"] == 97
    assert pooled["original_total"] == 300 and pooled["missing_original_count"] == 297
    assert seed["A_eval_mean"] is None and not seed["all_required_complete"]
    assert seed["conditional_A_eval_mean"] is not None


def test_missing_evidence_produces_failed_audit_without_generating_any_data():
    result = audit.audit_evidence("__no_stability_evidence_for_unit_test__")
    assert result["status"] == "failed" and result["mismatch_count"] > 0
    assert result["original_samples"] == 0
    assert not result["source_panels_sha256"]
    json.dumps(result, allow_nan=False)
