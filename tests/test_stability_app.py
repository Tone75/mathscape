"""Saved-view integration tests: no observations or bootstrap draws permitted."""

import base64
import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest

from mathscape import bootstrap, bootstrap_ui, inference, inference_ui, stability, stability_ui


APP = Path(__file__).resolve().parents[1] / "app.py"
MODE = "再標本化回数と端点の安定性"


@pytest.fixture(autouse=True)
def forbid_generation_and_duplicate_defaults(monkeypatch, caplog):
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-result view must never generate or resample")

    for module, names in [(stability, ["generate_sample", "generate_bootstrap_means", "bootstrap_means", "rng_for_panel"]),
                          (bootstrap, ["draw_replicate", "bootstrap_means"]), (bootstrap_ui, ["draw_replicate"]),
                          (inference, ["generate_samples"]), (inference_ui, ["generate_samples"])]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    caplog.set_level(logging.WARNING, logger=policies.__name__)
    yield
    assert not any("was created with a default value" in record.getMessage()
                   and "Session State API" in record.getMessage() for record in caplog.get_records("call"))


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def healthy(app):
    assert not app.exception and not app.error


def open_view():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    healthy(app)
    return app


def chart(app):
    return json.loads(app.get("plotly_chart")[0].proto.spec)


def array(value):
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
    return np.asarray(value)


def saved_panel(dist="normal", n=5, seed=0):
    with np.load(stability_ui.EVIDENCE / "panels" / f"{dist}_n{n}_seed{seed}.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def test_outcome_independent_default_exact_original_hash_and_fixed_alpha():
    app = open_view()
    panel = saved_panel()
    np.testing.assert_array_equal(array(chart(app)["data"][0]["x"]), panel["samples"][0])
    digest = hashlib.sha256(panel["samples"][0].astype("<f8").tobytes()).hexdigest()
    assert digest == panel["sample_sha256"][0]
    assert any(digest in caption.value for caption in app.caption)
    assert "原標本ID=0" in app.success[0].value
    assert "固定α=0.05" in app.success[0].value
    assert any("旧自由操作モードのαとは独立" in item.value for item in app.info)
    assert not any("有意水準" in item.label for item in app.selectbox)


@pytest.mark.parametrize("b,k", [(199, 0), (999, 9), (4999, 19)])
def test_B_and_k_preserve_original_and_use_exact_stored_corresponding_endpoints(b, k):
    app = open_view()
    original = array(chart(app)["data"][0]["x"]).copy()
    widget(app.selectbox, "保存bootstrap回数 B").set_value(b).run()
    widget(app.selectbox, "表示する内側系列 k").set_value(k).run()
    healthy(app)
    panel = saved_panel()
    bi = list(panel["B_values"]).index(b)
    spec = chart(app)
    np.testing.assert_array_equal(array(spec["data"][0]["x"]), original)
    for method, mi in (("percentile", 1), ("bca", 2)):
        name = stability_ui.METHODS[method]
        selected = next(trace for trace in spec["data"] if trace["name"] == f"{name} · 選択k={k}")
        np.testing.assert_array_equal(array(selected["x"]), [panel["lower"][0, bi, mi, k], panel["upper"][0, bi, mi, k]])
        all_series = next(trace for trace in spec["data"] if trace["name"] == f"{name} · 全K系列")
        for inner in range(20):
            assert all_series["x"][inner * 3] == panel["lower"][0, bi, mi, inner]
            assert all_series["x"][inner * 3 + 1] == panel["upper"][0, bi, mi, inner]
    diagnostics = app.dataframe[0].value
    row = diagnostics.loc[diagnostics.method == "bca"].iloc[0]
    for name in ("q_low", "q_high", "tau_low", "tau_high", "a", "z0"):
        assert row[name] == panel[name][0, bi, 2, k]


def test_method_changes_and_mode_lab_roundtrips_preserve_original_and_all_view_controls():
    app = open_view()
    widget(app.selectbox, "保存原標本ID").set_value(7).run()
    original = array(chart(app)["data"][0]["x"]).copy()
    widget(app.selectbox, "保存bootstrap回数 B").set_value(4999).run()
    widget(app.selectbox, "表示する内側系列 k").set_value(6).run()
    widget(app.multiselect, "安定性で表示する手法").set_value(["bca"]).run()
    widget(app.radio, "推論モード").set_value("信頼区間の方法比較").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    widget(app.radio, "Labを選択").set_value("Fractal Explorer").run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    healthy(app)
    assert widget(app.radio, "推論モード").value == MODE
    assert widget(app.selectbox, "保存原標本ID").value == 7
    assert widget(app.selectbox, "保存bootstrap回数 B").value == 4999
    assert widget(app.selectbox, "表示する内側系列 k").value == 6
    assert widget(app.multiselect, "安定性で表示する手法").value == ["bca"]
    np.testing.assert_array_equal(array(chart(app)["data"][0]["x"]), original)
    assert len(chart(app)["data"]) == 3
    widget(app.multiselect, "安定性で表示する手法").set_value([]).run()
    healthy(app)
    assert len(chart(app)["data"]) == 1


@pytest.mark.parametrize("dist,n", [("normal", 5), ("centered_exponential", 5), ("centered_exponential", 20)])
def test_required_saved_conditions_seed0_outer0_are_read_without_generation(dist, n):
    app = open_view()
    widget(app.selectbox, "保存母集団").set_value(dist).run()
    widget(app.selectbox, "保存標本数 n").set_value(n).run()
    healthy(app)
    np.testing.assert_array_equal(array(chart(app)["data"][0]["x"]), saved_panel(dist, n)["samples"][0])


def test_original_selection_changes_only_saved_data_and_retains_inner_selection():
    app = open_view()
    before = array(chart(app)["data"][0]["x"]).copy()
    widget(app.selectbox, "表示する内側系列 k").set_value(4).run()
    widget(app.selectbox, "保存master seed").set_value(2).run()
    widget(app.selectbox, "保存原標本ID").set_value(99).run()
    healthy(app)
    after = array(chart(app)["data"][0]["x"])
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(after, saved_panel(seed=2)["samples"][99])
    assert widget(app.selectbox, "表示する内側系列 k").value == 4


def test_exact_all36_pooled_csv_table_and_selected_seed100_distinguished_from_pool300():
    app = open_view()
    saved = pd.read_csv(stability_ui.EVIDENCE / "pooled_summary.csv")
    pd.testing.assert_frame_equal(app.dataframe[-1].value.reset_index(drop=True), saved, check_dtype=False)
    assert len(app.dataframe[-1].value) == 36
    assert set(app.dataframe[2].value.original_total) == {300}
    assert set(app.dataframe[3].value.original_total) == {100}
    assert any("独立単位は300原標本" in element.value for element in app.markdown)
    assert len(app.get("download_button")) == 1
    rows, payload = stability_ui._table(str(stability_ui.EVIDENCE / "pooled_summary.csv"),
                                         (stability_ui.EVIDENCE / "pooled_summary.csv").stat().st_mtime_ns)
    assert payload == (stability_ui.EVIDENCE / "pooled_summary.csv").read_bytes()
    assert len(rows) == 36


def test_selected_original_SD_A_warning_rates_match_saved_original_metrics_csv():
    app = open_view()
    saved = pd.read_csv(stability_ui.EVIDENCE / "original_metrics.csv")
    expected = saved.loc[(saved.distribution == "normal") & (saved.n == 5) & (saved.master_seed == 0) & (saved.outer_id == 0)]
    actual = app.dataframe[1].value
    for column in ("lower_sd", "upper_sd", "A", "C", "warning_rate", "se"):
        np.testing.assert_allclose(actual[column], expected[column], rtol=1e-13)
    assert any("真値や無限Bの正解ではありません" in element.value for element in app.info)
    assert any("端点の安定性と95%の被覆は別" in element.value for element in app.info)


def test_missing_saved_results_show_appropriate_error_without_resampling(monkeypatch):
    monkeypatch.setattr(stability_ui, "EVIDENCE", Path("__no_saved_stability_for_test__"))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    assert not app.exception
    assert len(app.error) == 1 and "保存結果を表示できません" in app.error[0].value


def test_fixed_alpha_does_not_inherit_old_free_exploration_alpha():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.selectbox, "有意水準 α").set_value(0.10).run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    healthy(app)
    assert "固定α=0.05" in app.success[0].value
    widget(app.radio, "推論モード").set_value("t推論の探索").run()
    assert widget(app.selectbox, "有意水準 α").value == 0.10


def test_original_hash_mismatch_is_visible_and_prevents_invalid_plot(monkeypatch):
    panel = saved_panel()
    panel["samples"][0, 0] += 1
    monkeypatch.setattr(stability_ui, "_panel", lambda *args: panel)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    assert not app.exception and len(app.error) == 1
    assert not app.get("plotly_chart")
