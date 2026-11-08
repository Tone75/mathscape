"""Real-app bootstrap mode contracts, using fixed deterministic UI inputs."""

import base64
from contextlib import nullcontext
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest

from mathscape import bootstrap, bootstrap_ui


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def duplicate_defaults_are_errors(monkeypatch, caplog):
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    caplog.set_level(logging.WARNING, logger=policies.__name__)
    yield
    assert not any("was created with a default value" in record.getMessage()
                   and "Session State API" in record.getMessage()
                   for record in caplog.get_records("call"))


@pytest.fixture(autouse=True)
def fixed_ui_draws(monkeypatch):
    """Test state/rendering without running stochastic development experiments.

    The production draw signature is bound on every call, so integration typos
    cannot be masked by a permissive fixture. Numerical engine validation is in
    test_bootstrap.py; these are fixed inputs, not scientific Monte Carlo data.
    """
    calls = []
    signature = inspect.signature(bootstrap.draw_replicate)

    def draw(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        config = bound.arguments.copy()
        calls.append(config)
        n = config["n"]
        shift = config["master_seed"] * 0.01 + config["replicate_id"] * 0.07
        sample = np.linspace(-1, 1, n) + shift
        if config["distribution"] == "centered_exponential":
            sample = np.linspace(0, 1, n) ** 3 - 0.3 + shift
        means = np.mean(sample) + np.linspace(-0.75, 0.75, config["B_boot"])
        return sample, means

    monkeypatch.setattr(bootstrap_ui, "draw_replicate", draw)
    return calls


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def healthy(app):
    assert not app.exception
    assert not app.error


def open_mode():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.radio, "推論モード").set_value("信頼区間の方法比較").run()
    healthy(app)
    return app


def execute(app):
    widget(app.button, "比較条件を適用して実行").click().run()
    healthy(app)
    return app


def state(app):
    return app.session_state[bootstrap_ui.STATE_KEY]


def plots(app):
    return [json.loads(element.proto.spec) for element in app.get("plotly_chart")]


def array(value):
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
    return np.asarray(value)


def test_default_mode_is_pending_and_explicit_draw_uses_correct_engine_arguments(fixed_ui_draws):
    app = open_mode()
    assert state(app)["applied"] is None
    assert len(fixed_ui_draws) == 0
    assert widget(app.selectbox, "比較する母集団").value == "centered_exponential"
    assert widget(app.selectbox, "比較標本数 n").value == 5
    assert widget(app.selectbox, "比較の有意水準 α").value == 0.05
    assert widget(app.number_input, "比較のmaster seed").value == 0
    assert widget(app.number_input, "比較の試行ID").value == 0
    execute(app)
    assert len(fixed_ui_draws) == 1
    assert fixed_ui_draws[0] == dict(distribution="centered_exponential", n=5, master_seed=0,
                                   replicate_id=0, B_boot=999, namespace=bootstrap.MAIN_NAMESPACE)
    assert state(app)["sample"].shape == (5,)
    assert state(app)["bootstrap_means"].shape == (999,)
    assert "試行ID=0" in app.success[0].value
    assert "B_boot=999" in app.success[0].value


def test_same_axis_contains_original_sample_and_all_three_interval_endpoints():
    app = execute(open_mode())
    chart = plots(app)[0]
    np.testing.assert_array_equal(array(chart["data"][0]["x"]), state(app)["sample"])
    assert len(chart["data"]) == 4
    for trace, method in zip(chart["data"][1:], bootstrap_ui.METHODS):
        np.testing.assert_allclose(array(trace["x"]), [state(app)["intervals"][method]["lower"],
                                                     state(app)["intervals"][method]["upper"]])
    np.testing.assert_array_equal(array(plots(app)[1]["data"][0]["x"]), state(app)["bootstrap_means"])
    assert any("再標本化しても元の観測情報は増えません" in element.value for element in app.info)
    assert any("95%の確率で存在するという意味ではありません" in element.value for element in app.markdown)
    assert not any("p値" in element.label or "検出力" in element.label for element in app.metric)


def test_alpha_only_reuses_original_sample_and_bootstrap_distribution_exactly(fixed_ui_draws):
    app = execute(open_mode())
    sample, means = state(app)["sample"].copy(), state(app)["bootstrap_means"].copy()
    lower = state(app)["intervals"]["percentile"]["lower"]
    widget(app.selectbox, "比較の有意水準 α").set_value(0.01).run()
    assert state(app)["applied"]["alpha"] == 0.05
    assert state(app)["intervals"]["percentile"]["lower"] == lower
    assert any("未適用" in item.value for item in app.warning)
    execute(app)
    assert len(fixed_ui_draws) == 1
    np.testing.assert_array_equal(state(app)["sample"], sample)
    np.testing.assert_array_equal(state(app)["bootstrap_means"], means)
    assert state(app)["intervals"]["percentile"]["lower"] < lower
    assert any("尾部" in item.value for item in app.warning)
    assert "α=0.01" in app.success[0].value


def test_method_display_changes_and_empty_selection_never_resample(fixed_ui_draws):
    app = execute(open_mode())
    means = state(app)["bootstrap_means"].copy()
    widget(app.multiselect, "表示する区間手法").set_value(["bca"]).run()
    assert len(plots(app)[0]["data"]) == 2
    assert plots(app)[0]["data"][1]["name"] == "BCa"
    widget(app.multiselect, "表示する区間手法").set_value([]).run()
    healthy(app)
    assert len(plots(app)[0]["data"]) == 1
    assert any("区間手法を表示していません" in item.value for item in app.info)
    assert len(fixed_ui_draws) == 1
    np.testing.assert_array_equal(state(app)["bootstrap_means"], means)


def test_mode_and_lab_roundtrip_preserve_pending_applied_and_selected_methods(fixed_ui_draws):
    app = execute(open_mode())
    sample, means = state(app)["sample"].copy(), state(app)["bootstrap_means"].copy()
    widget(app.selectbox, "比較標本数 n").set_value(20).run()
    widget(app.selectbox, "比較の有意水準 α").set_value(0.10).run()
    widget(app.multiselect, "表示する区間手法").set_value(["t", "bca"]).run()
    widget(app.radio, "推論モード").set_value("t推論の探索").run()
    widget(app.radio, "推論モード").set_value("信頼区間の方法比較").run()
    widget(app.radio, "Labを選択").set_value("Probability Explorer").run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    healthy(app)
    assert widget(app.radio, "推論モード").value == "信頼区間の方法比較"
    assert widget(app.selectbox, "比較標本数 n").value == 20
    assert widget(app.selectbox, "比較の有意水準 α").value == 0.10
    assert widget(app.multiselect, "表示する区間手法").value == ["t", "bca"]
    assert state(app)["applied"]["n"] == 5
    assert state(app)["applied"]["alpha"] == 0.05
    assert len(fixed_ui_draws) == 1
    np.testing.assert_array_equal(state(app)["sample"], sample)
    np.testing.assert_array_equal(state(app)["bootstrap_means"], means)


@pytest.mark.parametrize("dist,n", [("normal", 5), ("centered_exponential", 5), ("centered_exponential", 20)])
def test_three_required_population_conditions_execute_without_exceptions(dist, n):
    app = open_mode()
    widget(app.selectbox, "比較する母集団").set_value(dist).run()
    widget(app.selectbox, "比較標本数 n").set_value(n).run()
    execute(app)
    assert state(app)["applied"]["dist"] == dist
    assert state(app)["sample"].size == n
    assert all(record["valid"] for record in state(app)["intervals"].values())


def test_new_trial_changes_visible_id_with_same_applied_conditions(fixed_ui_draws):
    app = execute(open_mode())
    original = state(app)["sample"].copy()
    widget(app.button, "同じ適用条件で次の試行").click().run()
    healthy(app)
    assert len(fixed_ui_draws) == 2
    assert state(app)["applied"]["replicate_id"] == 1
    assert widget(app.number_input, "比較の試行ID").value == 1
    assert "試行ID=1" in app.success[0].value
    assert state(app)["pending"] == state(app)["applied"]
    assert not np.array_equal(state(app)["sample"], original)
    execute(app)
    assert len(fixed_ui_draws) == 2


def test_new_trial_preserves_different_unapplied_conditions(fixed_ui_draws):
    app = execute(open_mode())
    widget(app.selectbox, "比較標本数 n").set_value(20).run()
    widget(app.number_input, "比較のmaster seed").set_value(2).run()
    widget(app.button, "同じ適用条件で次の試行").click().run()
    healthy(app)
    assert state(app)["applied"]["n"] == 5
    assert state(app)["applied"]["master_seed"] == 0
    assert state(app)["applied"]["replicate_id"] == 1
    assert state(app)["pending"]["n"] == 20
    assert state(app)["pending"]["master_seed"] == 2
    assert any("未適用" in item.value for item in app.warning)
    assert len(fixed_ui_draws) == 2


def fake_evidence(monkeypatch):
    rows = []
    for dist in bootstrap_ui.DISTRIBUTIONS:
        for n in (5, 20, 100):
            for seed in (0, 1, 2):
                for method in bootstrap_ui.METHODS:
                    rows.append(dict(distribution=dist, n=str(n), master_seed=str(seed), method=method,
                                     R="2000", B_boot="999", alpha="0.05", total="2000", valid_count="2000",
                                     invalid_count="0", cover_count="1800", coverage="0.9", coverage_low="0.88",
                                     coverage_high="0.92", coverage_mcse="0.0067", miss_above_count="50",
                                     miss_above="0.025", miss_above_mcse="0.0035", miss_below_count="150",
                                     miss_below="0.075", miss_below_mcse="0.0059", mean_width="1.2",
                                     width_sd="0.3", width_median="1.1", width_mcse="0.0067", tail_warning_count="100"))
    monkeypatch.setattr(bootstrap_ui, "_load_summary", lambda: rows)

    class ExistingPath:
        def __truediv__(self, value):
            return self

        def exists(self):
            return True

    monkeypatch.setattr(bootstrap_ui, "EVIDENCE_DIR", ExistingPath())
    lower = np.tile(np.linspace(-1, -0.1, 60), (3, 1))
    upper = lower + 1.5
    valid = np.ones((3, 60), dtype=bool)
    valid[2, 4] = False
    data = dict(replicate_id=np.arange(60), methods=np.array(list(bootstrap_ui.METHODS)),
                lower=lower, upper=upper, valid=valid,
                reason=np.where(valid, "", "fixture invalid BCa"), tail_warning=np.zeros((3, 60), dtype=bool))
    load_calls = []

    def load(path, allow_pickle):
        load_calls.append(allow_pickle)
        return nullcontext(data)

    monkeypatch.setattr(bootstrap_ui.np, "load", load)
    return rows, data, load_calls


def test_saved_comparison_same_first50_ids_uses_all_r_and_all_seed_rows(monkeypatch, fixed_ui_draws):
    rows, saved, calls = fake_evidence(monkeypatch)
    app = execute(open_mode())
    fixed_plot = plots(app)[2]
    assert len(fixed_plot["data"]) == 3
    for method_index, method in enumerate(bootstrap_ui.METHODS):
        trace = fixed_plot["data"][method_index]
        valid_ids = np.flatnonzero(saved["valid"][method_index, :50])
        assert len(trace["x"]) == valid_ids.size * 3
        for line, replicate in enumerate(valid_ids):
            assert f"replicate ID={replicate}" in trace["text"][line * 3]
            assert trace["x"][line * 3] == saved["lower"][method_index, replicate]
            assert trace["x"][line * 3 + 1] == saved["upper"][method_index, replicate]
    table = app.dataframe[-1].value
    assert len(table) == 9
    assert set(table["master_seed"]) == {0, 1, 2}
    assert set(table["R"]) == {2000}
    assert set(table["coverage"]) == {0.9}
    assert set(table["B_boot"]) == {999}
    assert all(flag is False for flag in calls)
    assert len(fixed_ui_draws) == 1


def test_free_alpha_is_separate_from_fixed_alpha_and_display_seed_survives_roundtrip(monkeypatch, fixed_ui_draws):
    fake_evidence(monkeypatch)
    app = execute(open_mode())
    widget(app.selectbox, "比較の有意水準 α").set_value(0.10).run()
    execute(app)
    assert any("固定α=0.05" in item.value for item in app.info)
    assert set(app.dataframe[-1].value["alpha"]) == {0.05}
    widget(app.selectbox, "保存区間図のmaster seed").set_value(2).run()
    widget(app.radio, "推論モード").set_value("t推論の探索").run()
    widget(app.radio, "推論モード").set_value("信頼区間の方法比較").run()
    assert widget(app.selectbox, "保存区間図のmaster seed").value == 2
    assert len(fixed_ui_draws) == 1


def test_missing_fixed_evidence_does_not_trigger_outer_repetitions(monkeypatch, fixed_ui_draws):
    monkeypatch.setattr(bootstrap_ui, "_load_summary", lambda: [])
    app = execute(open_mode())
    assert any("固定実験結果はまだ保存されていません" in item.value for item in app.info)
    assert len(plots(app)) == 2
    assert len(fixed_ui_draws) == 1


def test_partial_fixed_results_are_explicitly_incomplete(monkeypatch):
    rows, _, _ = fake_evidence(monkeypatch)
    rows[:] = [row for row in rows if not (row["distribution"] == "centered_exponential"
                                          and row["n"] == "5" and row["master_seed"] == "2"
                                          and row["method"] == "bca")]
    app = execute(open_mode())
    assert any("固定比較は未完了" in item.value and "seed=2/BCa" in item.value for item in app.warning)
    assert len(app.dataframe[-1].value) == 8


def test_undefined_bca_is_reasoned_and_never_replaced_with_t(monkeypatch):
    original = bootstrap.intervals

    def invalid_bca(*args, **kwargs):
        results = original(*args, **kwargs)
        results["bca"] = results["bca"] | dict(valid=False, lower=np.nan, upper=np.nan, reason="fixture invalid acceleration")
        return results

    monkeypatch.setattr(bootstrap_ui, "intervals", invalid_bca)
    app = execute(open_mode())
    assert any("fixture invalid acceleration" in item.value for item in app.warning)
    assert [trace["name"] for trace in plots(app)[0]["data"]] == ["原標本の観測点", "t", "percentile bootstrap"]
    assert np.isnan(state(app)["intervals"]["bca"]["lower"])


def test_maximum_trial_disables_next_trial_without_overflow():
    app = open_mode()
    widget(app.number_input, "比較の試行ID").set_value(2**32 - 1).run()
    execute(app)
    assert widget(app.button, "同じ適用条件で次の試行").disabled
