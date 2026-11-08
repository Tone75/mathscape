"""State and numerical display contracts of the real four-Lab Streamlit app."""

import base64
import csv
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest

from mathscape import inference_ui
from mathscape.inference import generate_samples


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def reject_duplicate_widget_defaults(monkeypatch, caplog):
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    caplog.set_level(logging.WARNING, logger=policies.__name__)
    yield
    assert not any("was created with a default value" in record.getMessage()
                   and "Session State API" in record.getMessage()
                   for record in caplog.get_records("call"))


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def app_at_inference():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    assert_healthy(app)
    return app


def execute(app):
    widget(app.button, "条件を適用して実行").click().run()
    assert_healthy(app)
    return app


def test_existing_session_can_add_comparison_state_without_resampling():
    app = execute(app_at_inference())
    saved = experiment(app)["samples"].copy()
    experiment(app).pop("fixed_seed")
    app.run()
    assert_healthy(app)
    assert experiment(app)["fixed_seed"] == 0
    np.testing.assert_array_equal(experiment(app)["samples"], saved)


def assert_healthy(app):
    assert not app.exception
    assert not app.error


def experiment(app):
    return app.session_state[inference_ui.STATE_KEY]


def plots(app):
    return [json.loads(element.proto.spec) for element in app.get("plotly_chart")]


def array(values):
    if isinstance(values, dict) and "bdata" in values:
        result = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        if "shape" in values:
            result = result.reshape(tuple(int(size) for size in values["shape"].split(",")))
        return result
    return np.asarray(values)


def test_explicit_default_execution_and_first_sample_plot():
    app = app_at_inference()
    assert experiment(app)["applied"] is None
    assert not plots(app)
    assert widget(app.slider, "1標本の観測数 n").value == 20
    assert widget(app.slider, "真の標準化平均差 d").value == 0.0
    assert widget(app.selectbox, "有意水準 α").value == 0.05
    assert widget(app.number_input, "master seed").value == 0
    assert widget(app.number_input, "独立実験の反復数 B").value == 500
    execute(app)
    state = experiment(app)
    np.testing.assert_array_equal(state["samples"], generate_samples("normal", 20, 500, d=0, seed=0))
    assert state["samples"].shape == (500, 20)
    assert state["summary"]["denominator"] == 500
    sample_trace = plots(app)[0]["data"][1]
    np.testing.assert_array_equal(array(sample_trace["x"]), state["samples"][0])
    assert widget(app.metric, "第1標本の平均").value == f"{state['result']['mean'][0]:.5g}"
    assert widget(app.metric, "両側p値").value == f"{state['result']['p'][0]:.5g}"
    assert "試行ID=0" in app.success[0].value
    assert any("不棄却は効果がない" in text.value for text in app.markdown)


def test_alpha_changes_inference_only_after_apply_and_preserves_samples():
    app = execute(app_at_inference())
    samples = experiment(app)["samples"].copy()
    lower = experiment(app)["result"]["lower"].copy()
    p_values = experiment(app)["result"]["p"].copy()
    widget(app.selectbox, "有意水準 α").set_value(0.10).run()
    assert experiment(app)["applied"]["alpha"] == 0.05
    assert "α=0.05" in app.success[0].value
    assert any("未適用" in item.value for item in app.warning)
    np.testing.assert_array_equal(experiment(app)["samples"], samples)
    execute(app)
    assert experiment(app)["applied"]["alpha"] == 0.10
    np.testing.assert_array_equal(experiment(app)["samples"], samples)
    np.testing.assert_array_equal(experiment(app)["result"]["p"], p_values)
    assert np.all(experiment(app)["result"]["lower"] > lower)
    assert "α=0.1" in app.success[0].value


@pytest.mark.parametrize("label, distribution, n, d", [
    ("差がなくても棄却することがある", "normal", 20, 0.0),
    ("同じ真の差でもnで検出しやすさが変わる", "normal", 20, 0.5),
    ("母分布の歪み", "centered_exponential", 5, 0.0),
])
def test_three_observation_presets_are_pending_until_run(label, distribution, n, d):
    app = execute(app_at_inference())
    before = experiment(app)["samples"].copy()
    widget(app.selectbox, "推論の観察プリセット").set_value(label).run()
    assert experiment(app)["pending"]["distribution"] == distribution
    assert experiment(app)["pending"]["n"] == n
    assert experiment(app)["pending"]["d"] == d
    np.testing.assert_array_equal(experiment(app)["samples"], before)
    assert inference_ui.OBSERVATIONS[label] in [item.value for item in app.info]
    execute(app)
    assert experiment(app)["applied"]["distribution"] == distribution
    assert experiment(app)["applied"]["n"] == n
    assert experiment(app)["applied"]["d"] == d
    assert_healthy(app)


def test_lab_roundtrip_preserves_applied_samples_and_unapplied_controls():
    app = app_at_inference()
    widget(app.selectbox, "推論の観察プリセット").set_value("母分布の歪み").run()
    execute(app)
    samples = experiment(app)["samples"].copy()
    widget(app.slider, "真の標準化平均差 d").set_value(0.4).run()
    widget(app.selectbox, "有意水準 α").set_value(0.01).run()
    widget(app.radio, "Labを選択").set_value("Probability Explorer").run()
    widget(app.slider, "平均 μ").set_value(2.0).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    assert_healthy(app)
    assert widget(app.selectbox, "母集団の分布").value == "centered_exponential"
    assert widget(app.slider, "1標本の観測数 n").value == 5
    assert widget(app.slider, "真の標準化平均差 d").value == 0.4
    assert widget(app.selectbox, "有意水準 α").value == 0.01
    assert experiment(app)["applied"]["d"] == 0.0
    assert experiment(app)["applied"]["alpha"] == 0.05
    np.testing.assert_array_equal(experiment(app)["samples"], samples)
    assert "d=0" in app.success[0].value
    assert any("未適用" in item.value for item in app.warning)


def test_new_sample_increments_visible_trial_with_same_applied_conditions():
    app = execute(app_at_inference())
    original = experiment(app)["samples"].copy()
    widget(app.slider, "真の標準化平均差 d").set_value(0.7).run()
    widget(app.button, "適用条件で新しい標本").click().run()
    assert_healthy(app)
    state = experiment(app)
    assert state["generation"]["trial_id"] == 1
    assert state["applied"]["d"] == 0.0
    assert state["pending"]["d"] == 0.7
    assert not np.array_equal(original, state["samples"])
    np.testing.assert_array_equal(state["samples"], generate_samples("normal", 20, 500, trial_id=1))
    assert "試行ID=1" in app.success[0].value
    assert "seed=0" in app.success[0].value


def test_reapply_same_generation_does_not_resample_and_all_b_used():
    app = app_at_inference()
    widget(app.number_input, "独立実験の反復数 B").set_value(200).run()
    execute(app)
    before = experiment(app)["samples"].copy()
    execute(app)
    np.testing.assert_array_equal(experiment(app)["samples"], before)
    summary = experiment(app)["summary"]
    assert summary["denominator"] == 200
    intervals = plots(app)[1]["data"]
    displayed_lines = sum(len(trace["x"]) // 3 for trace in intervals)
    assert displayed_lines == 100
    result = experiment(app)["result"]
    assert summary["reject_count"] == np.count_nonzero(result["reject"])
    assert summary["cover_count"] == np.count_nonzero((result["lower"] <= 0) & (0 <= result["upper"]))


def test_maximum_n_and_b_remain_bounded_and_render():
    app = app_at_inference()
    widget(app.slider, "1標本の観測数 n").set_value(200).run()
    widget(app.number_input, "独立実験の反復数 B").set_value(5000).run()
    execute(app)
    assert experiment(app)["samples"].shape == (5000, 200)
    assert experiment(app)["samples"].size == 1_000_000
    assert experiment(app)["summary"]["denominator"] == 5000


def fake_fixed_file(monkeypatch):
    rows = []
    for seed in (0, 1, 2):
        for distribution in ("normal", "centered_exponential"):
            for n in (5, 20, 100):
                rows.append(dict(distribution=distribution, master_seed=seed, d=0.0, alpha=0.05,
                                 n=n, B=5000, denominator=5000, invalid_count=0,
                                 reject_rate=0.04 + seed * 0.01, reject_low=0.03 + seed * 0.01,
                                 reject_high=0.05 + seed * 0.01, coverage=0.95,
                                 coverage_low=0.94, coverage_high=0.96,
                                 mean_width=1 / np.sqrt(n), width_mcse=0.001, theory_power=0.05))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    # In-memory fixture, not a generated or rerun scientific experiment.
    monkeypatch.setattr(inference_ui, "SUMMARY_PATH", SimpleNamespace(
        exists=lambda: True, open=lambda **kwargs: io.StringIO(content)))


def test_fixed_comparison_uses_explicit_master_and_only_matching_conditions(monkeypatch):
    fake_fixed_file(monkeypatch)
    app = execute(app_at_inference())
    assert len(plots(app)) == 4
    fixed = plots(app)[2]
    np.testing.assert_array_equal(array(fixed["data"][0]["x"]), [5, 20, 100])
    np.testing.assert_allclose(array(fixed["data"][0]["y"]), 0.04)
    assert len(fixed["data"]) == 3  # normal theory is present
    widget(app.selectbox, "固定比較のmaster seed").set_value(2).run()
    np.testing.assert_allclose(array(plots(app)[2]["data"][0]["y"]), 0.06)
    widget(app.slider, "真の標準化平均差 d").set_value(0.3).run()
    execute(app)
    assert len(plots(app)) == 2
    assert any("一致する固定結果はありません" in item.value for item in app.info)
    widget(app.slider, "真の標準化平均差 d").set_value(0.0).run()
    widget(app.selectbox, "有意水準 α").set_value(0.10).run()
    execute(app)
    assert len(plots(app)) == 2
    assert any("近い条件の結果へ置き換えていません" in item.value for item in app.info)


def test_nonnormal_fixed_comparison_has_no_normal_theory_curve(monkeypatch):
    fake_fixed_file(monkeypatch)
    app = app_at_inference()
    widget(app.selectbox, "推論の観察プリセット").set_value("母分布の歪み").run()
    execute(app)
    assert len(plots(app)[2]["data"]) == 2
    assert any("厳密な理論検出力として流用しません" in item.value for item in app.info)


def test_fixed_master_selection_survives_lab_roundtrip(monkeypatch):
    fake_fixed_file(monkeypatch)
    app = execute(app_at_inference())
    widget(app.selectbox, "固定比較のmaster seed").set_value(2).run()
    widget(app.radio, "Labを選択").set_value("Chaos Explorer").run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    assert_healthy(app)
    assert widget(app.selectbox, "固定比較のmaster seed").value == 2
    np.testing.assert_allclose(array(plots(app)[2]["data"][0]["y"]), 0.06)


def test_smallest_sample_and_single_repetition_are_explicit():
    app = app_at_inference()
    widget(app.slider, "1標本の観測数 n").set_value(2).run()
    widget(app.number_input, "独立実験の反復数 B").set_value(1).run()
    execute(app)
    assert experiment(app)["samples"].shape == (1, 2)
    assert experiment(app)["summary"]["denominator"] == 1
    assert np.isnan(experiment(app)["summary"]["width_mcse"])
    assert any("平均幅のMonte Carlo標準誤差：未定義" in item.value for item in app.caption)


def test_missing_fixed_results_are_reported_without_substitution(monkeypatch):
    monkeypatch.setattr(inference_ui, "SUMMARY_PATH", SimpleNamespace(exists=lambda: False))
    app = execute(app_at_inference())
    assert len(plots(app)) == 2
    assert any("固定実験ファイルはまだありません" in item.value for item in app.info)


def test_zero_variance_samples_are_explicitly_invalid(monkeypatch):
    monkeypatch.setattr(inference_ui, "generate_samples",
                        lambda **kwargs: np.zeros((kwargs["repetitions"], kwargs["n"])))
    app = execute(app_at_inference())
    assert experiment(app)["summary"]["invalid_count"] == 500
    assert experiment(app)["summary"]["denominator"] == 500
    assert any("未定義" in item.value for item in app.warning)
    assert any("無効結果を含みます" in item.value for item in app.warning)
    assert np.isnan(experiment(app)["result"]["p"]).all()
