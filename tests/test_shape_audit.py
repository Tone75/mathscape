"""Fixed-array RQ-M04 independent audit references; no stochastic runs."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from scripts import audit_shape as audit


def fixed_panel(r=3, k=10):
    samples = np.array([np.linspace(-2, 2, 20) * (1 + i / 10) + i / 5 for i in range(r)])
    shape = (r, 2, 3, k)
    panel = {name: np.full(shape, np.nan) for name in
             ("lower", "upper", "a", "z0", "q_low", "q_high", "tau_low", "tau_high")}
    panel.update(samples=samples, sample_sha256=np.array([audit.sample_hash(sample) for sample in samples]),
                 outer_id=np.arange(r), inner_id=np.arange(k), B_values=np.array(audit.BS), methods=np.array(audit.METHODS),
                 distribution=np.array("gamma_0.5"), n=np.array(20), master_seed=np.array(0), R=np.array(r), K=np.array(k),
                 alpha=np.array(0.05), namespace=np.array(audit.MAIN_NAMESPACE), valid=np.ones(shape, dtype=bool),
                 reason=np.full(shape, "", dtype="U40"), tail_warning=np.zeros(shape, dtype=bool))
    for i, sample in enumerate(samples):
        half = stats.t.ppf(0.975, 19) * np.std(sample, ddof=1) / np.sqrt(20)
        panel["lower"][i, :, 0] = np.mean(sample) - half
        panel["upper"][i, :, 0] = np.mean(sample) + half
        for bi, b in enumerate(audit.BS):
            for mi in (1, 2):
                panel["lower"][i, bi, mi] = -0.8 + np.arange(k) / 100 + i / 10
                panel["upper"][i, bi, mi] = panel["lower"][i, bi, mi] + 1.6
                panel["q_low"][i, bi, mi] = 0.025
                panel["q_high"][i, bi, mi] = 0.975
                panel["tau_low"][i, bi, mi] = panel["tau_high"][i, bi, mi] = b * 0.025
                if mi == 2:
                    panel["a"][i, bi, mi] = panel["z0"][i, bi, mi] = 0
    return panel


def test_auditor_has_no_primary_or_runner_imports():
    tree = ast.parse(Path(audit.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ("mathscape" in name or "run_shape" in name or "audit_stability" in name) for name in imports)


def test_panel_contract_hash_se_t_a_tau_q_ids_and_namespace():
    panel = fixed_panel()
    assert audit.inspect_panel(panel, strict_design=False) == []
    assert any("R400/n20/K10" in message for message in audit.inspect_panel(panel))


@pytest.mark.parametrize("field,index,value,fragment", [
    ("sample_sha256", 0, "0" * 64, "hash mismatch"),
    ("lower", (0, 1, 0, 0), -99, "t reference mismatch"),
    ("a", (0, 1, 2, 0), 0.1, "acceleration differs"),
    ("z0", (0, 1, 2, 0), 0.1234, "tie-count lattice"),
    ("q_high", (0, 1, 2, 0), 0.025, "quantile range"),
    ("tau_low", (0, 1, 2, 0), 99, "lower tau differs"),
    ("tail_warning", (0, 0, 2, 0), True, "warning mask differs"),
    ("inner_id", 0, 9, "inner IDs"),
])
def test_saved_panel_corruption_is_detected(field, index, value, fragment):
    panel = fixed_panel()
    panel[field][index] = value
    assert any(fragment in message for message in audit.inspect_panel(panel, strict_design=False))


def test_directional_misses_and_stability_scales_have_hand_computed_values():
    panel = fixed_panel(r=1, k=4)
    panel["lower"][0, 0, 2] = [-3, -1, 0, 1]
    panel["upper"][0, 0, 2] = [-1, 1, 2, 3]
    row = next(row for row in audit.original_metrics(panel) if row["B"] == 999 and row["method"] == "bca")
    assert row["C"] == 0.5
    assert row["Mbelow"] == 0.25 and row["Mabove"] == 0.25 and row["T"] == 0
    assert row["C"] + row["Mbelow"] + row["Mabove"] == 1
    assert row["lower_sd"] == pytest.approx(np.sqrt(35 / 12))
    assert row["S"] == pytest.approx(np.sqrt(35 / 12))
    assert row["Ase"] == pytest.approx(row["S"] / row["se"])
    assert row["Apop"] == pytest.approx(row["S"] * np.sqrt(20))
    panel["upper"][0, 0, 2, 1] = -0.5
    panel["lower"][0, 0, 2, 1] = -2.5
    row = next(row for row in audit.original_metrics(panel) if row["B"] == 999 and row["method"] == "bca")
    assert row["Mbelow"] == 0.5 and row["T"] == 0.25  # U<0 increases positive T


def test_original_cluster_mcse_t_counts_and_unclipped_intervals():
    rows = [row for row in audit.original_metrics(fixed_panel()) if row["B"] == 999 and row["method"] == "bca"]
    for row, value in zip(rows, [0, 0.5, 1]):
        row["C"] = value
    summary = audit.summarize_originals(rows, expected=3)
    assert summary["C"] == 0.5
    assert summary["C_mcse"] == pytest.approx(0.5 / np.sqrt(3))
    assert summary["C_decision_low"] < 0 and summary["C_decision_high"] > 1
    t_rows = [row for row in audit.original_metrics(fixed_panel()) if row["B"] == 999 and row["method"] == "t"]
    t_summary = audit.summarize_originals(t_rows, expected=3)
    assert t_summary["interval_count"] == t_summary["original_total"] == 3
    assert t_summary["S"] == t_summary["Ase"] == t_summary["Apop"] == 0


def test_invalid_or_missing_original_preserves_total_and_primary_NA():
    panel = fixed_panel()
    panel["valid"][0, 0, 2, 1] = False
    panel["lower"][0, 0, 2, 1] = panel["upper"][0, 0, 2, 1] = np.nan
    rows = [row for row in audit.original_metrics(panel) if row["B"] == 999 and row["method"] == "bca"]
    summary = audit.summarize_originals(rows, expected=4)
    assert summary["original_total"] == 4 and summary["missing_original_count"] == 1
    assert summary["interval_count"] == 30 and summary["invalid_count"] == 1
    assert summary["C"] is None and summary["T"] is None
    assert summary["conditional_original_count"] == 2 and summary["conditional_C"] is not None


def test_fractional_constant_original_is_rejected_before_roundoff_variance():
    panel = fixed_panel(r=1)
    panel["samples"][0] = np.full(20, 0.8)
    panel["sample_sha256"][0] = audit.sample_hash(panel["samples"][0])
    panel["se"] = np.array([np.nan])
    panel["valid"][:] = False
    panel["reason"][:] = "sample_constant"
    panel["tail_warning"][:] = False
    for field in ("lower", "upper", "a", "z0", "q_low", "q_high", "tau_low", "tau_high"):
        panel[field][:] = np.nan
    assert audit.inspect_panel(panel, strict_design=False) == []
    rows = audit.original_metrics(panel)
    assert all(not row["complete"] and np.isnan(row["se"]) for row in rows)
    assert all(row["C"] is None and row["T"] is None and row["Ase"] is None for row in rows)
    for method, count in (("t", 1), ("percentile", 10), ("bca", 10)):
        selected = [row for row in rows if row["method"] == method and row["B"] == 4999]
        summary = audit.summarize_originals(selected, expected=1)
        assert summary["original_total"] == 1 and summary["complete_original_count"] == 0
        assert summary["interval_count"] == summary["invalid_count"] == count
        assert summary["C"] is None and summary["T"] is None
    panel["valid"][0, 0, 0, 0] = True
    panel["reason"][0, 0, 0, 0] = ""
    assert any("t validity differs" in error for error in audit.inspect_panel(panel, strict_design=False))


def confirmation_fixture():
    rows = []
    for seed in (0, 1, 2):
        for outer in range(400):
            wave = (outer % 11 - 5) * 0.002
            for dist in ("gamma_0.5", "gamma_16"):
                for b in (999, 4999):
                    rows.append(dict(distribution=dist, master_seed=seed, outer_id=outer, B=b, method="bca", complete=True,
                                     C=0.9 + wave if dist == "gamma_16" else (0.89 if seed == 0 else 0.80) + wave,
                                     T=(0.01 if seed == 0 else 0.10) + wave))
    return rows


def test_confirmations_use_independent_population_Welch_and_seed_zero_thresholds():
    rows = confirmation_fixture()
    quantities = audit.confirmation(rows)
    assert quantities["H1"]["status"] == quantities["H2"]["status"] == "supported"
    assert quantities["H1"]["interval"]["N1"] == quantities["H1"]["interval"]["N0"] == 1200
    assert quantities["H2"]["interval"]["N"] == 1200
    assert -0.02 < quantities["H1"]["seed_points"][0] < 0  # seed threshold is zero, not -0.02
    assert 0 < quantities["H2"]["seed_points"][0] < 0.03  # seed threshold is zero, not +0.03
    # The within-ID coverage differences here are nearly constant, but an
    # independent population comparison must retain both group variances.
    assert quantities["H1"]["interval"]["mcse"] > 0


@pytest.mark.parametrize("dist,b,expect_h1,expect_h2", [
    ("normal", 4999, "supported", "supported"),
    ("gamma_0.5", 999, "supported", "supported"),
    ("gamma_16", 4999, "inconclusive", "supported"),
    ("gamma_0.5", 4999, "inconclusive", "inconclusive"),
])
def test_required_comparison_local_invalid_gates(dist, b, expect_h1, expect_h2):
    rows = confirmation_fixture()
    match = next((row for row in rows if row["distribution"] == dist and row["B"] == b), None)
    if match:
        match.update(complete=False, C=None, T=None)
    else:
        rows.append(dict(distribution=dist, master_seed=0, outer_id=0, B=b, method="bca", complete=False, C=None, T=None))
    result = audit.confirmation(rows)
    assert result["H1"]["status"] == expect_h1
    assert result["H2"]["status"] == expect_h2


@pytest.mark.parametrize("low,high,threshold,direction,seeds,complete,expected", [
    (-0.04, -0.03, -0.02, "less", [-0.01] * 3, True, "supported"),
    (-0.02, -0.01, -0.02, "less", [-0.01] * 3, True, "inconclusive"),
    (-0.01, 0.01, -0.02, "less", [0.0] * 3, True, "not_supported"),
    (0.04, 0.08, 0.03, "greater", [0.01] * 3, True, "supported"),
    (0.04, 0.08, 0.03, "greater", [0.01, 0, 0.01], True, "inconclusive"),
    (0.01, 0.02, 0.03, "greater", [0.01] * 3, True, "not_supported"),
    (0.01, 0.02, 0.03, "greater", [0.01] * 3, False, "inconclusive"),
])
def test_preregistered_decisions_boundary_seed_and_invalid_gates(low, high, threshold, direction, seeds, complete, expected):
    assert audit.quantity_status(dict(low=low, high=high, defined=True), threshold, direction, seeds, complete=complete) == expected


def paired_fixture():
    rows = []
    for seed in (0, 1, 2):
        for outer in range(400):
            for b in (999, 4999):
                for mi, method in enumerate(audit.METHODS):
                    value = (outer % 2) * 0.25 + (0.25 if b == 4999 else 0) + mi * 0.125
                    rows.append(dict(distribution="normal", master_seed=seed, outer_id=outer, B=b, method=method, complete=True,
                                     C=value, Ase=value, Apop=2 * value, T=value, mean_width=1 + value))
    return rows


def test_B_descriptive_difference_is_paired_on_originals_with_original_level_mcse():
    rows = paired_fixture()
    result = audit.paired_summaries(rows, "B")
    assert len(result) == 72
    row = next(row for row in result if row["distribution"] == "normal" and row["master_seed"] == "pooled" and row["method"] == "bca")
    assert row["original_total"] == row["C_complete_original_count"] == 1200
    assert row["C_difference"] == 0.25 and row["C_sd"] == row["C_mcse"] == 0
    assert row["C_decision_low"] is None  # constant paired difference, not an independent-groups SE
    assert row["Apop_difference"] == 0.5


def test_method_descriptive_difference_pairs_same_original_B_and_retains_invalids():
    rows = paired_fixture()
    result = audit.paired_summaries(rows, "method")
    assert len(result) == 144
    row = next(row for row in result if row["distribution"] == "normal" and row["master_seed"] == 0
               and row["B"] == 4999 and row["method"] == "bca" and row["reference"] == "percentile")
    assert row["C_difference"] == 0.125 and row["C_mcse"] == 0
    next(item for item in rows if item["master_seed"] == item["outer_id"] == 0 and item["B"] == 4999 and item["method"] == "bca")["complete"] = False
    result = audit.paired_summaries(rows, "method")
    row = next(row for row in result if row["distribution"] == "normal" and row["master_seed"] == 0
               and row["B"] == 4999 and row["method"] == "bca" and row["reference"] == "percentile")
    assert row["original_total"] == 400 and row["C_complete_original_count"] == 399
    assert row["C_difference"] is None
    assert row["conditional_C_difference"] == 0.125


def test_hypothesis_reference_has_two_required_quantities_N1200_and_no_nonfinite_json():
    reference = audit.hypothesis_reference(confirmation_fixture())
    assert reference["confirmatory_quantities"] == 2
    assert reference["quantities"]["H1"]["independent_groups"]
    assert reference["quantities"]["H1"]["N1"] == reference["quantities"]["H1"]["N0"] == 1200
    assert reference["quantities"]["H2"]["N"] == 1200
    assert len(reference["seed_details"]) == 3
    json.dumps(reference, allow_nan=False)


def test_missing_evidence_fails_without_generating_or_importing_primary_sources():
    result = audit.audit_evidence("__no_shape_evidence_for_test__")
    assert result["status"] == "failed" and result["mismatch_count"] > 0
    assert result["original_samples"] == 0 and not result["source_panels_sha256"]
    assert result["table_counts"] == {}
    json.dumps(result, allow_nan=False)
