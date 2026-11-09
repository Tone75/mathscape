"""Fixed-array M06 audit references; no RNG, smoke or main execution."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import special, stats

from scripts import audit_matched_stability as audit


def fixed_panel(r=3, k=30):
    samples = np.array([np.linspace(-0.5, 2, 20) + i / 10 for i in range(r)])
    shape = (r, 2, 3, k)
    panel = {key: np.full(shape, np.nan) for key in ("lower", "upper", "a", "z0", "q_low", "q_high", "tau_low", "tau_high")}
    panel.update(samples=samples, sample_sha256=np.array([audit.sample_hash(sample).encode() for sample in samples], dtype="S64"),
                 outer_id=np.arange(r, dtype=np.uint16), inner_id=np.arange(k, dtype=np.uint8), B_values=np.array(audit.BS),
                 methods=np.array(audit.METHODS), distribution=np.array("gamma"), master_seed=np.array(0), n=np.array(20),
                 R=np.array(r), K=np.array(k), alpha=np.array(0.05), namespace=np.array(audit.MAIN_NAMESPACE),
                 valid=np.ones(shape, dtype=bool), tail_warning=np.zeros(shape, dtype=bool), reason_codes=np.zeros(shape, dtype=np.uint8),
                 reason_labels=np.array(["", "sample_constant"], dtype="U40"))
    for i, sample in enumerate(samples):
        half = stats.t.ppf(0.975, 19) * np.std(sample, ddof=1) / np.sqrt(20)
        panel["lower"][i, :, 0] = sample.mean() - half
        panel["upper"][i, :, 0] = sample.mean() + half
        for bi, b in enumerate(audit.BS):
            for mi in (1, 2):
                panel["lower"][i, bi, mi] = -0.8 + np.arange(k) * 0.01
                panel["upper"][i, bi, mi] = panel["lower"][i, bi, mi] + 1.6
                panel["q_low"][i, bi, mi], panel["q_high"][i, bi, mi] = 0.025, 0.975
                panel["tau_low"][i, bi, mi] = panel["tau_high"][i, bi, mi] = b * 0.025
                if mi == 2:
                    panel["a"][i, bi, mi] = panel["z0"][i, bi, mi] = 0
    return panel


def test_auditor_imports_no_primary_model_runner_or_previous_auditor():
    tree = ast.parse(Path(audit.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(module and ("mathscape" in module or "run_matched" in module or "audit_matched" in module) for module in imports)


def test_panel_hashes_common_ids_Bs_t_a_and_diagnostics():
    panel = fixed_panel()
    assert audit.inspect_panel(panel, strict_design=False) == []
    assert any("R200/K30" in error for error in audit.inspect_panel(panel))
    panel["inner_id"][0] = 29
    assert any("inner IDs" in error for error in audit.inspect_panel(panel, strict_design=False))


@pytest.mark.parametrize("field,index,value,fragment", [
    ("sample_sha256", 0, b"0" * 64, "hash mismatch"),
    ("lower", (0, 1, 0, 0), -99, "t reference differs"),
    ("a", (0, 1, 2, 0), 0.1, "acceleration differs"),
    ("z0", (0, 1, 2, 0), 0.1234, "rank grid"),
    ("q_high", (0, 1, 2, 0), 0.025, "invalid quantiles"),
    ("tau_low", (0, 1, 2, 0), 99, "tau differs"),
    ("tail_warning", (0, 0, 2, 0), True, "warning differs"),
])
def test_saved_diagnostic_corruption_detected(field, index, value, fragment):
    panel = fixed_panel()
    panel[field][index] = value
    assert any(fragment in error for error in audit.inspect_panel(panel, strict_design=False))


def test_endpoint_SD_S_Ase_Apop_Vse_partition_and_t_zero_control():
    panel = fixed_panel(r=1, k=4)
    panel["lower"][0, 0, 2] = [-2, -1, 0, 1]
    panel["upper"][0, 0, 2] = [0, 1, 2, 3]
    rows = audit.original_metrics(panel)
    row = next(row for row in rows if row["method"] == "bca" and row["B"] == 999)
    assert row["lower_sd"] == row["upper_sd"] == pytest.approx(np.sqrt(5 / 3))
    assert row["S"] == pytest.approx(np.sqrt(5 / 3))
    assert row["Ase"] == pytest.approx(row["S"] / row["se"])
    assert row["Apop"] == pytest.approx(row["S"] * np.sqrt(20))
    assert row["Vse"] == pytest.approx(row["Ase"] ** 2)
    assert row["C"] == 0.75 and row["Mabove"] == 0.25 and row["Mbelow"] == 0
    assert row["T"] == -0.25 and row["C"] + row["Mabove"] + row["Mbelow"] == 1
    for row in rows:
        if row["method"] == "t":
            assert row["total_inner"] == 1
            assert row["lower_sd"] == row["upper_sd"] == row["S"] == row["Ase"] == row["Apop"] == row["Vse"] == 0


def test_invalid_K_member_keeps_denominators_primary_NA_and_conditional_count():
    panel = fixed_panel()
    panel["valid"][0, 0, 2, 0] = False
    panel["lower"][0, 0, 2, 0] = panel["upper"][0, 0, 2, 0] = np.nan
    rows = [row for row in audit.original_metrics(panel) if row["method"] == "bca" and row["B"] == 999]
    result = audit.summarize_originals(rows, expected=4)
    assert result["original_total"] == 4 and result["missing_original_count"] == 1
    assert result["interval_count"] == 90 and result["invalid_count"] == 1
    assert result["complete_original_count"] == result["conditional_original_count"] == 2
    assert result["Ase"] is None and result["C"] is None
    assert result["conditional_Ase"] is not None


def test_linear_quantiles_and_original_unit_MCSE_not_RK():
    stats_result = audit._metric_stats([1, 2, 3])
    assert stats_result["_p05"] == pytest.approx(1.1)
    assert stats_result["_p25"] == 1.5 and stats_result["_p50"] == 2
    assert stats_result["_p75"] == 2.5 and stats_result["_p95"] == pytest.approx(2.9)
    assert stats_result["_mcse"] == pytest.approx(1 / np.sqrt(3))


@pytest.mark.parametrize("low,high,defined,complete,status", [
    (-0.2, -0.1, True, True, "supported"), (0, 0.1, True, True, "not_supported"),
    (0.1, 0.2, True, True, "not_supported"), (-0.1, 0, True, True, "inconclusive"),
    (-0.1, 0.1, True, True, "inconclusive"), (0, 0, False, True, "inconclusive"),
    (-0.2, -0.1, True, False, "inconclusive"),
])
def test_H2_strict_reduction_and_low_zero_boundary(low, high, defined, complete, status):
    assert audit.reduction_decision(dict(low=low, high=high, defined=defined), complete) == status


@pytest.mark.parametrize("statuses,expected", [
    (["supported", "supported"], "supported"), (["supported", "inconclusive"], "inconclusive"),
    (["not_supported", "inconclusive"], "not_supported"), (["supported", "not_supported"], "not_supported"),
])
def test_H2_joint_preserves_clear_refutation_alongside_missing(statuses, expected):
    assert audit.combine_reduction(statuses) == expected


@pytest.mark.parametrize("low,high,status,equivalent", [
    (0.011, 0.02, "supported", False), (-0.02, -0.011, "supported", False),
    (-0.009, 0.009, "not_supported", True), (-0.01, 0.009, "inconclusive", False),
    (-0.009, 0.01, "inconclusive", False),
])
def test_H1_absolute_SE_unit_margin_contact_not_pp(low, high, status, equivalent):
    result = audit.practical_decision(dict(low=low, high=high, defined=True))
    assert result["status"] == status and result["equivalence_supported"] == equivalent


def test_fixed_resampling_reference_half_ties_acceleration_and_linear_quantiles():
    sample = np.linspace(-0.5, 2, 20)
    mean = sample.mean()
    means = mean + np.linspace(-1, 1, 999)
    result = audit.reference_intervals(sample, means)
    expected = np.quantile(means, [0.025, 0.975], method="linear")
    np.testing.assert_allclose([result["percentile"]["lower"], result["percentile"]["upper"]], expected)
    probability = (np.count_nonzero(means < mean) + 0.5 * np.count_nonzero(means == mean)) / 999
    assert result["bca"]["z0"] == special.ndtri(probability)
    jack = (sample.sum() - sample) / 19
    centered = jack.mean() - jack
    assert result["bca"]["a"] == pytest.approx(np.sum(centered**3) / (6 * np.sum(centered**2) ** 1.5))
    assert not any(record["valid"] for record in audit.reference_intervals(np.full(20, 0.8), means).values())


def test_old_M05_K1_stability_NA_is_unchanged():
    rows = audit._csv(audit.ROOT / "evidence" / "rq_m05" / "pooled_summary.csv")
    assert all(np.isnan(float(row[field])) for row in rows for field in ("lower_sd", "upper_sd", "S", "Ase", "Apop"))


def confirmation_rows():
    rows = []
    for dist in audit.DISTRIBUTIONS:
        for seed in (0, 1, 2):
            for outer in range(200):
                low = 0.1 + (outer % 7) * 0.005 + (0.02 if dist == "lognormal" else 0)
                high = 0.05 + (outer % 5) * 0.0002 + (0.003 if dist == "lognormal" else 0)
                for b, value in ((999, low), (4999, high)):
                    rows.append(dict(distribution=dist, master_seed=seed, outer_id=outer, B=b, method="bca", complete=True, Ase=value))
    return rows


def test_confirmation_uses_600_originals_paired_D_and_ratio_of_means():
    rows = confirmation_rows()
    result = audit.confirmation(rows)
    assert result["H1"]["equivalence_supported"] and result["H1"]["interval"]["N1"] == 600
    assert result["H2"]["status"] == "supported"
    low = audit._vector(rows, "gamma", "pooled", 999, "bca", "Ase")
    high = audit._vector(rows, "gamma", "pooled", 4999, "bca", "Ase")
    quantity = result["H2_gamma"]
    assert quantity["interval"]["N"] == 600
    assert quantity["interval"]["mean"] == pytest.approx(np.mean(high - .75 * low))
    assert quantity["ratio_of_means"] == pytest.approx(high.mean() / low.mean())
    assert not np.isclose(quantity["ratio_of_means"], np.mean(high / low))


def test_unrelated_B_or_population_invalid_only_gates_needed_primary_quantity():
    rows = confirmation_rows()
    row = next(row for row in rows if row["distribution"] == "gamma" and row["master_seed"] == row["outer_id"] == 0 and row["B"] == 999)
    row.update(complete=False, Ase=None)
    result = audit.confirmation(rows)
    assert result["H1"]["equivalence_supported"]  # B4999 remains complete
    assert result["H2_gamma"]["status"] == "inconclusive"
    assert result["H2_gamma"]["interval"]["N"] == 600 and result["H2_gamma"]["interval"]["finite_count"] == 599
    assert result["H2_lognormal"]["status"] == "supported"
    assert result["H2"]["status"] == "inconclusive"


def test_fixed_designated_stream_reconstruction_detects_wrong_mean_and_B_z0(tmp_path):
    panel = fixed_panel(r=1)
    indices = np.full((4999, 20), 19, dtype=np.uint8)
    indices[:900] = 0
    indices[999:2999] = 0
    means = panel["samples"][0][indices].mean(axis=1)
    for bi, b in enumerate(audit.BS):
        results = audit.reference_intervals(panel["samples"][0], means[:b])
        for mi, method in enumerate(audit.METHODS):
            for key, value in results[method].items():
                panel[key][0, bi, mi, 0] = value
    assert panel["z0"][0, 0, 2, 0] != panel["z0"][0, 1, 2, 0]
    fields = dict(indices=indices, means=means, distribution="gamma", master_seed=0, outer_id=0, inner_id=0, n=20,
                  B=4999, B_values=np.array(audit.BS), K=30, alpha=.05, namespace=audit.MAIN_NAMESPACE,
                  native_indices_dtype="int64", stored_indices_dtype="uint8", sample_sha256=panel["sample_sha256"][0])
    path = tmp_path / "fixed_stream.npz"
    np.savez_compressed(path, **fields)
    assert audit.audit_stream(path, panel, "gamma", 0, 0) == []
    panel["z0"][0, 0, 2, 0] = panel["z0"][0, 1, 2, 0]
    assert any("z0 differs" in message for message in audit.audit_stream(path, panel, "gamma", 0, 0))
    means[0] += 1
    np.savez_compressed(path, **fields)
    assert any("saved bootstrap means differ" in message for message in audit.audit_stream(path, panel, "gamma", 0, 0))


def test_full_saved_fixture_all_inputs_streams_summaries_contrasts_and_hypotheses(tmp_path, monkeypatch):
    """Use accepted saved inputs and deterministic identity resamples, no RNG.

    Reference inference is evaluated once per original and broadcast solely for
    schema testing. This is not an independent experiment or a performance run.
    """
    from scripts import run_matched_stability_benchmark as reference
    from mathscape import matched_stability, matched_moments

    def forbidden(*args, **kwargs):
        raise AssertionError("saved-fixture test may not generate or execute experiments")

    for module, names in [(reference, ["run_benchmark", "run_smoke", "run_cell", "generate_bootstrap_means"]),
                          (matched_stability, ["generate_bootstrap_means", "rng_for_panel"]),
                          (matched_moments, ["generate_sample", "generate_bootstrap_means", "rng_for_panel"])]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    manifest = reference.make_manifest()
    saved_inputs = reference.load_inputs(manifest["inputs"], root=reference.ROOT)
    target = tmp_path / "saved_fixture"
    (target / "panels").mkdir(parents=True)
    (target / "streams").mkdir()
    all_rows = []
    identity_indices = np.broadcast_to(np.arange(20, dtype=np.uint8), (4999, 20))
    for dist, seed in reference.fixed_cells():
        saved = saved_inputs[(dist, seed)]
        panel = reference._allocate(saved, 200, 30, (999, 4999))
        labels = [""]
        for oi, sample in enumerate(saved["samples"]):
            context = reference.prepare_sample(sample, alpha=.05)
            means = np.mean(sample[identity_indices], axis=1)
            records = reference.infer_prefixes(context, means, B_values=(999, 4999))
            for bi, b in enumerate((999, 4999)):
                for mi, method in enumerate(audit.METHODS):
                    record = records[b][method]
                    for key in reference.DIAGNOSTICS + ("valid", "tail_warning"):
                        panel[key][oi, bi, mi] = record[key]
                    if record["reason"] not in labels:
                        labels.append(record["reason"])
                    panel["reason_codes"][oi, bi, mi] = labels.index(record["reason"])
            original_data = {key: panel[key][oi] for key in reference.DIAGNOSTICS + ("valid", "tail_warning")}
            metrics = reference.original_metrics(original_data, float(context["se"]), n=20)
            for bi, b in enumerate((999, 4999)):
                for mi, method in enumerate(audit.METHODS):
                    row = dict(distribution=dist, master_seed=seed, outer_id=oi, n=20, R=200, K=30, B=b, alpha=.05,
                               method=method, se=float(context["se"]), sample_sha256=audit.sample_hash(sample))
                    row.update({key: values[bi, mi].item() for key, values in metrics.items()})
                    all_rows.append(row)
            if oi == 0:
                for inner in (0, 1, 29):
                    np.savez_compressed(target / "streams" / f"{dist}_seed{seed}_outer0_inner{inner}.npz",
                                        indices=identity_indices, means=means, distribution=dist, master_seed=seed, outer_id=0, inner_id=inner,
                                        n=20, B=4999, B_values=np.array(audit.BS), K=30, alpha=.05, namespace=audit.MAIN_NAMESPACE,
                                        native_indices_dtype="int64", stored_indices_dtype="uint8", sample_sha256=saved["sample_sha256"][0])
        panel["reason_labels"] = np.array(labels, dtype="U96")
        reference.save_panel(target / "panels" / f"{dist}_seed{seed}.npz", panel, dist, seed, 30, (999, 4999))
    for filename, rows in [("original_metrics.csv", all_rows), ("summary.csv", reference.summarize(all_rows)),
                           ("pooled_summary.csv", reference.summarize(all_rows, pooled=True)),
                           ("population_contrasts.csv", reference.population_contrasts(all_rows)),
                           ("paired_B_summary.csv", reference.paired_contrasts(all_rows, "B")),
                           ("paired_method_summary.csv", reference.paired_contrasts(all_rows, "method"))]:
        reference.write_csv(target / filename, rows)
    reference._write_json(target / "hypotheses.json", reference.assemble_hypotheses(all_rows))
    reference._write_json(target / "manifest.json", manifest)
    reference._write_json(target / "resources.json", dict(status="completed", completed_cells=6, incomplete_cells=[]))
    result = audit.audit_evidence(target)
    assert result["mismatches"] == []
    assert result["status"] == "passed"
    assert result["original_samples"] == 1200 and result["fully_reconstructed_streams"] == 18
    for name, count in [("original_metrics.csv", 7200), ("summary.csv", 36), ("pooled_summary.csv", 12),
                        ("population_contrasts.csv", 24), ("paired_B_summary.csv", 24), ("paired_method_summary.csv", 48)]:
        assert result["table_counts"][name] == dict(reconstructed=count, saved=count)
    json.dumps(result, allow_nan=False)
