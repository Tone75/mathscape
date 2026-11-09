"""M06 saved viewer: no model calls, exact arrays, stable axes and provenance."""

import base64
import csv
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

from mathscape import (bootstrap, bootstrap_ui, distribution_shape, inference, inference_ui,
                       matched_moments, matched_stability, matched_stability_ui, stability)
from scripts import run_matched_stability_benchmark


APP = Path(__file__).resolve().parents[1] / "app.py"
MODE = "同じ標本でも端点は揺れる"


@pytest.fixture(autouse=True)
def no_model_generation_or_experiments(monkeypatch, caplog):
    def forbidden(*args, **kwargs):
        raise AssertionError("saved M06 viewer may only read/plot evidence")

    for module, names in [(matched_stability, ["rng_for_panel", "generate_bootstrap_means", "prepare_sample", "infer_prefixes", "original_metrics", "make_inputs_manifest", "load_inputs"]),
                          (matched_moments, ["rng_for_panel", "generate_sample", "generate_bootstrap_means", "bootstrap_means", "prepare_sample", "infer_single"]),
                          (distribution_shape, ["rng_for_panel", "generate_sample", "generate_bootstrap_means", "bootstrap_means"]),
                          (stability, ["rng_for_panel", "generate_sample", "generate_bootstrap_means", "bootstrap_means", "prepare_sample", "infer_prefixes"]),
                          (bootstrap, ["draw_replicate", "bootstrap_means", "intervals"]), (bootstrap_ui, ["draw_replicate", "intervals"]),
                          (inference, ["generate_samples", "t_inference"]), (inference_ui, ["generate_samples", "t_inference"]),
                          (run_matched_stability_benchmark, ["run_cell", "run_benchmark", "run_smoke", "generate_bootstrap_means"])]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    caplog.set_level(logging.WARNING, logger=policies.__name__)
    yield
    assert not any("was created with a default value" in record.getMessage() and "Session State API" in record.getMessage()
                   for record in caplog.get_records("call"))


def widget(elements, label):
    return next(item for item in elements if item.label == label)


def healthy(app):
    assert not app.exception and not app.error


def open_view():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    healthy(app)
    return app


def plots(app):
    return [json.loads(item.proto.spec) for item in app.get("plotly_chart")]


def array(value):
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
    return np.asarray(value)


def original(app):
    return array(plots(app)[0]["data"][0]["x"])


def panel(dist="gamma", seed=0):
    with np.load(matched_stability_ui.EVIDENCE / "panels" / f"{dist}_seed{seed}.npz", allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def test_default_saved_example_equals_accepted_M05_and_fixed_metadata():
    app = open_view()
    saved = panel()
    np.testing.assert_array_equal(original(app), saved["samples"][0])
    digest = hashlib.sha256(original(app).astype("<f8").tobytes()).hexdigest()
    assert digest == saved["sample_sha256"][0].decode("ascii")
    with np.load(matched_stability_ui.M05 / "panels" / "gamma_seed0.npz", allow_pickle=False) as m05:
        np.testing.assert_array_equal(original(app), m05["samples"][0])
        assert digest == m05["sample_sha256"][0].decode("ascii")
    assert any(digest in item.value and "M05保存hashと一致" in item.value for item in app.caption)
    assert "固定α=.05・K30" in app.success[0].value
    assert widget(app.selectbox, "端点変動のbootstrap回数 B").value == 4999
    assert widget(app.selectbox, "端点変動の内側系列 k").value == 0
    assert plots(app)[0]["data"][0]["marker"]["symbol"] == "circle"


@pytest.mark.parametrize("dist", ["gamma", "lognormal"])
def test_B_k_and_method_visibility_keep_original_and_constant_all_B_axis(dist):
    app = open_view()
    widget(app.selectbox, "端点変動の母集団").set_value(dist).run()
    saved = panel(dist)
    before, axis = original(app).copy(), plots(app)[0]["layout"]["xaxis"]["range"]
    all_x = np.concatenate([saved["samples"][0], saved["lower"][0].ravel(), saved["upper"][0].ravel()])
    all_x = all_x[np.isfinite(all_x)]
    padding = max((all_x.max() - all_x.min()) * .05, 1e-6)
    np.testing.assert_array_equal(axis, [all_x.min() - padding, all_x.max() + padding])
    widget(app.selectbox, "端点変動のbootstrap回数 B").set_value(999).run()
    widget(app.selectbox, "端点変動の内側系列 k").set_value(29).run()
    widget(app.multiselect, "端点変動で表示する手法").set_value(["bca"]).run()
    healthy(app)
    np.testing.assert_array_equal(original(app), before)
    assert plots(app)[0]["layout"]["xaxis"]["range"] == axis
    selected = next(trace for trace in plots(app)[0]["data"] if trace["name"] == "BCa · 選択k=29")
    np.testing.assert_array_equal(array(selected["x"]), [saved["lower"][0, 0, 2, 29], saved["upper"][0, 0, 2, 29]])
    assert selected["marker"]["line"]["width"] >= 2
    widget(app.multiselect, "端点変動で表示する手法").set_value([]).run()
    healthy(app)
    assert plots(app)[0]["layout"]["xaxis"]["range"] == axis
    assert len(plots(app)[0]["data"]) == 1


def test_population_specific_seed_outer_k_restore_and_mode_Lab_hash_retention():
    app = open_view()
    widget(app.selectbox, "端点変動のmaster seed").set_value(2).run()
    widget(app.number_input, "端点変動の原標本ID").set_value(19).run()
    widget(app.selectbox, "端点変動の内側系列 k").set_value(8).run()
    gamma = original(app).copy()
    widget(app.selectbox, "端点変動の母集団").set_value("lognormal").run()
    assert widget(app.selectbox, "端点変動のmaster seed").value == 0
    assert widget(app.number_input, "端点変動の原標本ID").value == 0
    assert widget(app.selectbox, "端点変動の内側系列 k").value == 0
    widget(app.number_input, "端点変動の原標本ID").set_value(199).run()
    widget(app.selectbox, "端点変動の内側系列 k").set_value(24).run()
    widget(app.selectbox, "端点変動のbootstrap回数 B").set_value(999).run()
    widget(app.multiselect, "端点変動で表示する手法").set_value(["bca"]).run()
    lognormal = original(app).copy()
    assert not np.array_equal(gamma, lognormal)
    widget(app.radio, "推論モード").set_value("同じ歪度、違う分布").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    widget(app.radio, "Labを選択").set_value("Probability Explorer").run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    healthy(app)
    assert widget(app.radio, "推論モード").value == MODE
    assert widget(app.number_input, "端点変動の原標本ID").value == 199
    assert widget(app.selectbox, "端点変動の内側系列 k").value == 24
    assert widget(app.selectbox, "端点変動のbootstrap回数 B").value == 999
    assert widget(app.multiselect, "端点変動で表示する手法").value == ["bca"]
    np.testing.assert_array_equal(original(app), lognormal)
    widget(app.selectbox, "端点変動の母集団").set_value("gamma").run()
    assert widget(app.selectbox, "端点変動のmaster seed").value == 2
    assert widget(app.number_input, "端点変動の原標本ID").value == 19
    assert widget(app.selectbox, "端点変動の内側系列 k").value == 8
    np.testing.assert_array_equal(original(app), gamma)


def test_raw_SD_S_SE_Ase_Apop_and_diagnostics_match_saved_original_metrics():
    app = open_view()
    saved = pd.read_csv(matched_stability_ui.EVIDENCE / "original_metrics.csv")
    rows = saved[(saved.distribution == "gamma") & (saved.master_seed == 0) & (saved.outer_id == 0)]
    current = rows[(rows.method == "bca") & (rows.B == 4999)].iloc[0]
    for label, key in [("下端点SD", "lower_sd"), ("上端点SD", "upper_sd"), ("合成端点変動 S", "S"),
                       ("標本SE（s/√20）", "se"), ("原標本Ase", "Ase"), ("原標本Apop", "Apop")]:
        assert widget(app.metric, label).value == format(current[key], ".6g")
    for key in ("lower_sd", "upper_sd", "S", "se", "Ase", "Apop", "Vse", "C", "T"):
        np.testing.assert_allclose(app.dataframe[1].value[key], rows[key], rtol=1e-13)
    diagnostic = app.dataframe[0].value.loc[app.dataframe[0].value.method == "bca"].iloc[0]
    source = panel()
    for key in ("a", "z0", "q_low", "q_high", "tau_low", "tau_high"):
        assert diagnostic[key] == source[key][0, 1, 2, 0]


def test_all12_pooled_rows_and_MC_bars_equal_saved_CSV_and_t_counts_R_only():
    app = open_view()
    expected = pd.read_csv(matched_stability_ui.EVIDENCE / "pooled_summary.csv")
    actual = app.dataframe[-2].value
    pd.testing.assert_frame_equal(actual.astype(object).where(actual.notna(), None),
                                  expected.astype(object).where(expected.notna(), None), check_dtype=False)
    assert len(actual) == 12 and set(actual.original_total) == {600}
    assert set(actual.loc[actual.method == "t", "interval_count"]) == {600}
    assert set(actual.loc[actual.method != "t", "interval_count"]) == {600 * 30}
    for trace, dist in zip(plots(app)[1]["data"], matched_stability_ui.LABELS):
        rows = expected[(expected.distribution == dist) & (expected.method == "bca")].set_index("B").loc[[999, 4999]]
        np.testing.assert_allclose(array(trace["y"]), rows.Ase, rtol=1e-13)
        np.testing.assert_allclose(array(trace["error_y"]["array"]), rows.Ase_decision_high - rows.Ase, rtol=1e-12)
    assert any("内側乱数だけの区間ではありません" in item.value for item in app.info)


def test_H1_absolute_SE_unit_equivalence_band_and_H2_ratios_exact_saved_values():
    app = open_view()
    saved = json.loads((matched_stability_ui.EVIDENCE / "hypotheses.json").read_text())
    h1 = saved["quantities"]["H1"]
    chart = plots(app)[2]
    assert chart["data"][0]["x"] == [h1["difference"]]
    assert chart["data"][0]["error_x"]["arrayminus"] == [h1["difference"] - h1["low"]]
    band = next(shape for shape in chart["layout"]["shapes"] if shape["type"] == "rect")
    assert band["x0"] == -.01 and band["x1"] == .01
    assert widget(app.metric, "H1 平均Ase差 L−G").value == format(h1["difference"], "+.7f")
    assert any("被覆率の1ppではありません" in item.value for item in app.info)
    for trace, dist in zip(plots(app)[3]["data"], matched_stability_ui.LABELS):
        row = saved["quantities"][f"H2_{dist}"]
        assert trace["x"] == [row["mean"]]
        assert trace["error_x"]["array"] == [row["high"] - row["mean"]]
        assert widget(app.metric, f"{dist} 平均A4999/平均A999").value == format(row["ratio_mean_Ase_4999_over_999"], ".6f")
    assert any("両分布で平均の端点変動が25%超低減" in item.value for item in app.success)


def test_M05_K1_remains_NA_and_is_not_replaced_by_M06_t_zero_variation():
    path = matched_stability_ui.M05 / "pooled_summary.csv"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    app = open_view()
    old = app.dataframe[-1].value
    assert set(old.K) == {1}
    assert old[["lower_sd", "upper_sd", "S", "Ase", "Apop"]].isna().all().all()
    local = app.dataframe[1].value
    assert (local.loc[local.method == "t", ["lower_sd", "upper_sd", "S", "Ase", "Apop"]] == 0).all().all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_all_downloads_exact_summary_HTML_original_hash_and180_endpoint_records(monkeypatch):
    captured = {}
    actual_download = matched_stability_ui.st.download_button

    def capture(label, data, **kwargs):
        captured[kwargs["file_name"]] = data
        return actual_download(label, data, **kwargs)

    monkeypatch.setattr(matched_stability_ui.st, "download_button", capture)
    app = open_view()
    assert captured["rq_m06_pooled_summary.csv"] == (matched_stability_ui.EVIDENCE / "pooled_summary.csv").read_bytes()
    assert captured["rq_m06_overview.html"] == (matched_stability_ui.EVIDENCE / "figures" / "overview.html").read_bytes()
    assert len(captured) == 4 and all(isinstance(payload, bytes) and len(payload) > 0 for payload in captured.values())
    sample_rows = list(csv.DictReader(io.StringIO(captured["rq_m06_gamma_seed0_outer0_sample.csv"].decode())))
    values = np.array([float(row["X"]) for row in sample_rows], dtype="<f8")
    np.testing.assert_array_equal(values, original(app))
    assert hashlib.sha256(values.tobytes()).hexdigest() == sample_rows[0]["sample_sha256"]
    records = list(csv.DictReader(io.StringIO(captured["rq_m06_gamma_seed0_outer0_endpoints.csv"].decode())))
    assert len(records) == 180
    saved = panel()
    for record in records:
        bi, mi, k = [999, 4999].index(int(record["B"])), list(matched_stability_ui.METHODS).index(record["method"]), int(record["inner_id"])
        assert float(record["lower"]) == saved["lower"][0, bi, mi, k]
        assert float(record["upper"]) == saved["upper"][0, bi, mi, k]


def test_fixed_alpha_independent_from_free_exploration_controls():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.selectbox, "有意水準 α").set_value(.10).run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    healthy(app)
    assert "固定α=.05・K30" in app.success[0].value
    widget(app.radio, "推論モード").set_value("t推論の探索").run()
    assert widget(app.selectbox, "有意水準 α").value == .10


def test_M05_source_hash_mismatch_stops_view_without_inference(monkeypatch):
    monkeypatch.setattr(matched_stability_ui, "_old_hashes", lambda *args: np.full(10000, "0" * 64))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    assert not app.exception and len(app.error) == 1
    assert not app.get("plotly_chart")
