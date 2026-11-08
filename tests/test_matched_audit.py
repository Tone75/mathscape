"""Deterministic compact-data audit contracts; no resampling or experiments."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from scripts import audit_matched as audit


def fixed_panel(r=3):
    samples = np.array([np.linspace(-0.5, 2, 20) + i / 10 for i in range(r)])
    shape = (r, 3)
    panel = {key: np.full(shape, np.nan) for key in ("lower", "upper", "z0", "a", "q_low", "q_high", "tau_low", "tau_high")}
    panel.update(samples=samples, sample_sha256=np.array([audit.sample_hash(sample).encode() for sample in samples], dtype="S64"),
                 outer_id=np.arange(r, dtype=np.uint16), methods=np.array(audit.METHODS), valid=np.ones(shape, dtype=bool),
                 tail_warning=np.zeros(shape, dtype=bool), reason_codes=np.zeros(shape, dtype=np.uint8),
                 reason_labels=np.array(["", "sample_constant"], dtype="U40"),
                 n=np.array(20), R=np.array(r), K=np.array(1), B=np.array(4999), alpha=np.array(0.05),
                 namespace=np.array(audit.MAIN_NAMESPACE), distribution=np.array("gamma"), master_seed=np.array(0))
    for i, sample in enumerate(samples):
        half = stats.t.ppf(0.975, 19) * np.std(sample, ddof=1) / np.sqrt(20)
        panel["lower"][i, 0], panel["upper"][i, 0] = np.mean(sample) - half, np.mean(sample) + half
        for mi in (1, 2):
            panel["lower"][i, mi], panel["upper"][i, mi] = -0.8, 0.8
            panel["q_low"][i, mi], panel["q_high"][i, mi] = 0.025, 0.975
            panel["tau_low"][i, mi] = panel["tau_high"][i, mi] = 4999 * 0.025
            if mi == 2:
                panel["a"][i, mi] = panel["z0"][i, mi] = 0
    return panel


def test_independent_auditor_imports_no_primary_or_other_auditor():
    tree = ast.parse(Path(audit.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(module and ("mathscape" in module or "run_matched" in module or "audit_shape" in module) for module in modules)


def test_compact_panel_contract_and_byte_hashes():
    panel = fixed_panel()
    assert audit.inspect_panel(panel, strict_design=False) == []
    assert any("10000" in message for message in audit.inspect_panel(panel))
    metrics = audit.original_metrics(panel)
    assert metrics["complete"].all()
    for field in audit.STABILITY:
        assert np.isnan(metrics[field]).all()  # includes t: K1 never estimates conditional SD as zero


@pytest.mark.parametrize("field,index,value,fragment", [
    ("sample_sha256", 0, b"0" * 64, "hash mismatch"),
    ("lower", (0, 0), -99, "t reference differs"),
    ("a", (0, 2), 0.1, "acceleration differs"),
    ("z0", (0, 2), 0.1234, "rank lattice"),
    ("q_high", (0, 2), 0.025, "quantiles invalid"),
    ("tau_low", (0, 2), 99, "tau differs"),
    ("tail_warning", (0, 2), True, "warning differs"),
    ("outer_id", 0, 2, "original IDs"),
])
def test_compact_saved_corruption_is_detected(field, index, value, fragment):
    panel = fixed_panel()
    panel[field][index] = value
    assert any(fragment in message for message in audit.inspect_panel(panel, strict_design=False))


def test_reason_dictionary_out_of_range_and_invalid_interval_contract():
    panel = fixed_panel()
    panel["reason_codes"][0, 2] = 99
    with pytest.raises(ValueError, match="dictionary"):
        audit.reasons(panel)
    panel["reason_codes"][0, 2] = 1
    assert any("valid/reason" in message for message in audit.inspect_panel(panel, strict_design=False))


def test_C_T_signs_are_indicators_and_invalids_never_noncoverage():
    panel = fixed_panel()
    panel["lower"][0, 2], panel["upper"][0, 2] = -2, -1
    panel["lower"][1, 2], panel["upper"][1, 2] = 1, 2
    panel["lower"][2, 2], panel["upper"][2, 2] = -1, 1
    values = audit.original_metrics(panel)
    np.testing.assert_array_equal(values["C"][:, 2], [0, 0, 1])
    np.testing.assert_array_equal(values["Mbelow"][:, 2], [1, 0, 0])
    np.testing.assert_array_equal(values["Mabove"][:, 2], [0, 1, 0])
    np.testing.assert_array_equal(values["T"][:, 2], [1, -1, 0])
    np.testing.assert_array_equal(values["C"] + values["Mbelow"] + values["Mabove"], np.ones((3, 3)))
    panel["valid"][0, 2] = False
    values = audit.original_metrics(panel)
    assert np.isnan(values["C"][0, 2]) and np.isnan(values["T"][0, 2])
    assert values["total_inner"][0, 2] == values["invalid_inner"][0, 2] == 1


def test_fractional_constant_original_is_early_rejected_with_SE_NA():
    panel = fixed_panel(r=1)
    panel["samples"][0] = 0.8
    panel["sample_sha256"][0] = audit.sample_hash(panel["samples"][0]).encode()
    panel["valid"][:] = False
    panel["reason_codes"][:] = 1
    for field in ("lower", "upper", "a", "z0", "q_low", "q_high", "tau_low", "tau_high"):
        panel[field][:] = np.nan
    assert audit.inspect_panel(panel, strict_design=False) == []
    values = audit.original_metrics(panel)
    assert np.isnan(values["se"]).all() and not values["complete"].any()
    assert values["invalid_inner"].sum() == 3


@pytest.mark.parametrize("low,high,defined,complete,status,direction,equivalent", [
    (0.011, 0.02, True, True, "supported", "positive", False),
    (-0.02, -0.011, True, True, "supported", "negative", False),
    (-0.009, 0.009, True, True, "not_supported", "within_margin", True),
    (0.001, 0.009, True, True, "not_supported", "within_margin", True),
    (-0.01, 0.009, True, True, "inconclusive", "undetermined", False),
    (-0.009, 0.01, True, True, "inconclusive", "undetermined", False),
    (0.01, 0.02, True, True, "inconclusive", "undetermined", False),
    (-0.02, -0.01, True, True, "inconclusive", "undetermined", False),
    (-0.02, 0.02, True, True, "inconclusive", "undetermined", False),
    (0.011, 0.02, True, False, "inconclusive", "undetermined", False),
    (-0.009, 0.009, False, True, "inconclusive", "undetermined", False),
])
def test_strict_difference_and_equivalence_bounds(low, high, defined, complete, status, direction, equivalent):
    result = audit.practical_decision(dict(low=low, high=high, defined=defined), complete=complete)
    assert result == dict(status=status, direction=direction, equivalence_supported=equivalent)


def test_zero_variance_CI_is_undefined_and_MC_rate_interval_not_clipped():
    interval = audit.welch_interval(np.full(100, 0.8), np.full(100, 0.7))
    assert not interval["defined"]
    assert audit.practical_decision(interval)["status"] == "inconclusive"
    interval = audit.mean_interval([0, 0, 1])
    assert interval["low"] < 0 and interval["high"] > 1


def confirmation_panels():
    panels = {}
    for dist in audit.DISTRIBUTIONS:
        for seed in (0, 1, 2):
            cover = 9000 if dist == "gamma" else (8900 if seed == 0 else 9500)
            c = np.zeros((10000, 3)); c[:cover] = 1
            t = np.zeros((10000, 3))
            positive = (10000 - cover) // 2 if dist == "gamma" else (900 if seed == 0 else 400)
            t[cover:cover + positive] = 1
            t[cover + positive:] = -1
            values = dict(C=c, T=t, mean_width=np.tile(np.linspace(1, 2, 10000)[:, None], (1, 3)))
            panels[(dist, seed)] = (dict(outer_id=np.arange(10000)), values)
    return panels


def test_full_N30000_Welch_sign_L_minus_G_and_no_seed_gate():
    panels = confirmation_panels()
    result = audit.confirmation(panels)
    assert result["H1"]["status"] == result["H2"]["status"] == "supported"
    assert result["H1"]["direction"] == "positive"
    assert result["H1"]["interval"]["N1"] == result["H1"]["interval"]["N0"] == 30000
    assert result["H1"]["interval"]["difference"] > 0
    seed_zero = audit.population_intervals(panels, "bca", 0)["C"]
    assert seed_zero["difference"] < 0  # descriptive disagreement does not gate pooled decision


def test_one_required_missing_value_holds_primary_but_other_method_invalid_does_not():
    panels = confirmation_panels()
    panels[("lognormal", 0)][1]["C"][0, 0] = np.nan
    assert audit.confirmation(panels)["H1"]["status"] == "supported"  # t only
    panels[("lognormal", 0)][1]["C"][0, 2] = np.nan
    result = audit.confirmation(panels)
    assert result["H1"]["status"] == "inconclusive"
    assert result["H1"]["interval"]["N1"] == 30000
    assert result["H1"]["interval"]["finite_count1"] == 29999
    assert result["H2"]["status"] == "supported"


def test_full_saved_fixed_fixture_matches_runner_all_tables_and_manifest_schema(tmp_path, monkeypatch):
    """No random draws: repeat three fixed arrays to exercise the full schema.

    The reference writer/aggregator is called only by this integration test;
    audit_matched itself imports neither that runner nor its numerical model.
    This is a serialization/independent-reconstruction fixture, not an experiment.
    """
    from scripts import run_matched_benchmark as reference

    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic fixture must not generate/resample")

    for name in ("generate_sample", "generate_bootstrap_means", "run_benchmark", "run_smoke"):
        monkeypatch.setattr(reference, name, forbidden)
    target = tmp_path / "fixed_saved_fixture"
    (target / "panels").mkdir(parents=True)
    base = fixed_panel()
    base["lower"][:, 2] = [-2, 1, -1]
    base["upper"][:, 2] = [-1, 2, 1]
    indices = np.arange(10000) % 3
    fields = ("samples", "sample_sha256", "lower", "upper", "z0", "a", "q_low", "q_high", "tau_low", "tau_high", "valid", "tail_warning", "reason_codes")
    cells = {}
    with reference.metric_writer(target / "original_metrics.csv.gz") as writer:
        for dist in audit.DISTRIBUTIONS:
            for seed in (0, 1, 2):
                panel = {field: base[field][indices].copy() for field in fields}
                panel.update(outer_id=np.arange(10000, dtype=np.uint16), reason_labels=base["reason_labels"])
                metrics = reference.panel_metrics(panel)
                cells[(dist, seed)] = metrics
                reference.save_panel(target / "panels" / f"{dist}_seed{seed}.npz", panel, dist, seed, 10000, 4999, audit.MAIN_NAMESPACE)
                reference.write_metric_rows(writer, dist, seed, panel, metrics)
    for filename, rows in [("summary.csv", reference.summarize(cells)), ("pooled_summary.csv", reference.summarize(cells, pooled=True)),
                           ("population_contrasts.csv", reference.population_contrasts(cells)),
                           ("paired_method_contrasts.csv", reference.method_contrasts(cells))]:
        reference.write_csv(target / filename, rows)
    reference._write_json(target / "hypotheses.json", reference.assemble_hypotheses(cells))
    reference._write_json(target / "resources.json", dict(status="completed", completed_cells=6, incomplete_cells=[]))
    reference._write_json(target / "manifest.json", reference.make_manifest())
    result = audit.audit_evidence(target)
    assert result["mismatches"] == []
    assert result["status"] == "passed" and result["original_samples"] == 60000
    assert result["table_counts"]["original_metrics.csv.gz"] == dict(reconstructed=180000, saved=180000)
    assert result["table_counts"]["summary.csv"] == dict(reconstructed=18, saved=18)
    assert result["table_counts"]["pooled_summary.csv"] == dict(reconstructed=6, saved=6)
    assert result["table_counts"]["population_contrasts.csv"] == dict(reconstructed=12, saved=12)
    assert result["table_counts"]["paired_method_contrasts.csv"] == dict(reconstructed=24, saved=24)
    json.dumps(result, allow_nan=False)
