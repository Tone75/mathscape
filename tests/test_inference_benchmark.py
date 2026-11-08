"""Tiny deterministic runner checks; never execute the fixed 54-cell design."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.run_inference_benchmark import (
    BudgetExceeded, DESIGN, POLICY, conservative_count_band, enforce_budget,
    fixed_cells, run_cell, validate_manifest,
)


def test_fixed_design_counts_and_no_duplicate_keys():
    cells = fixed_cells()
    assert len(cells) == len(set(cells)) == 54
    assert sum(n * DESIGN["B"] for _, n, _, _ in cells) == 11250000
    assert len(cells) * DESIGN["B"] == 270000


def test_manifest_rejects_changed_design_and_limits():
    manifest = {"schema_version": 1, "question_id": "RQ-M01", "design": deepcopy(DESIGN),
                "policy": deepcopy(POLICY), "totals": {"cells": 54, "sample_groups": 270000, "observations": 11250000}}
    validate_manifest(manifest)
    manifest["design"]["B"] = 6000
    with pytest.raises(ValueError):
        validate_manifest(manifest)
    manifest["design"] = deepcopy(DESIGN)
    manifest["policy"]["wall_seconds"] = 301
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_conservative_bands_contain_nominal_and_handle_endpoints():
    low, high = conservative_count_band(5000, .05)
    assert 0 <= low < 250 < high <= 5000
    assert conservative_count_band(100, 0) == (0, 0)
    assert conservative_count_band(100, 1) == (100, 100)


@pytest.mark.parametrize("metric,value", [("wall", 300), ("rss", 2 * 1024**3), ("output", 50 * 1024**2)])
def test_each_resource_limit_stops(metric, value):
    values = {"wall": 1, "rss": 1, "output": 1}
    values[metric] = value
    with pytest.raises(BudgetExceeded):
        enforce_budget(values["wall"], values["rss"], values["output"])


def test_under_resource_limits_allowed():
    enforce_budget(299, 2 * 1024**3 - 1, 50 * 1024**2 - 1)


@pytest.mark.parametrize("distribution", ["normal", "centered_exponential"])
def test_tiny_cell_batching_preserves_result(distribution):
    whole = run_cell(distribution, 5, .2, 1, B=12, batch_size=12)
    split = run_cell(distribution, 5, .2, 1, B=12, batch_size=5)
    assert whole.keys() == split.keys()
    for name in whole:
        if isinstance(whole[name], (float, np.floating)) and np.isnan(whole[name]):
            assert np.isnan(split[name])
        else:
            assert whole[name] == split[name]
    assert split["denominator"] == 12
    assert split["invalid_count"] == 0
    if distribution == "normal":
        assert np.isfinite(split["theory_power"])
    else:
        assert np.isnan(split["theory_power"])
        assert split["reject_investigate"] == "not_applicable"


@pytest.mark.parametrize("B,batch", [(1, 1), (5001, 1), (5, 1001), (5, 0), (True, 1), (5, True)])
def test_tiny_cell_rejects_invalid_bounds(B, batch):
    with pytest.raises(ValueError):
        run_cell("normal", 5, 0, 0, B=B, batch_size=batch)


def test_budget_checked_before_each_batch_and_after_last():
    calls = []
    run_cell("normal", 5, 0, 0, check_budget=lambda: calls.append(1), B=12, batch_size=5)
    assert len(calls) == 4
