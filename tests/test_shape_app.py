"""RQ-M04 saved-view integration; prohibit all project generation APIs."""

import base64
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest

from mathscape import (bootstrap, bootstrap_ui, distribution_shape, inference,
                       inference_ui, shape_ui, stability)


APP = Path(__file__).resolve().parents[1] / "app.py"
MODE = "分布の形と被覆"


@pytest.fixture(autouse=True)
def no_generation_or_duplicate_defaults(monkeypatch, caplog):
    def forbidden(*args, **kwargs):
        raise AssertionError("saved shape viewer attempted observation/bootstrap generation")

    for module, names in [(distribution_shape, ["rng_for_panel", "generate_sample", "generate_bootstrap_means", "bootstrap_means"]),
                          (stability, ["rng_for_panel", "generate_sample", "generate_bootstrap_means", "bootstrap_means"]),
                          (bootstrap, ["draw_replicate", "bootstrap_means"]), (bootstrap_ui, ["draw_replicate"]),
                          (inference, ["generate_samples"]), (inference_ui, ["generate_samples"])]:
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


def panel(dist="normal", seed=0):
    with np.load(shape_ui.EVIDENCE / "panels" / f"{dist}_seed{seed}.npz", allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def original(app):
    return array(plots(app)[1]["data"][0]["x"])


def test_default_example_is_saved_normal_seed0_outer0_k0_and_fixed_metadata():
    app = open_view()
    saved = panel()
    np.testing.assert_array_equal(original(app), saved["samples"][0])
    digest = hashlib.sha256(original(app).astype("<f8").tobytes()).hexdigest()
    assert any(digest in item.value for item in app.caption)
    assert digest == saved["sample_sha256"][0]
    assert widget(app.selectbox, "形状ビューのbootstrap回数 B").value == 4999
    assert "n20・固定α=0.05" in app.success[0].value
    assert any("母平均0・母分散1・n=20" in item.value for item in app.caption)
    assert plots(app)[1]["data"][0]["marker"]["symbol"] == "circle"


@pytest.mark.parametrize("dist", ["gamma_0.5", "gamma_1", "gamma_16", "normal"])
def test_required_populations_exact_saved_original_default_ID_and_theory_pdf(dist):
    app = open_view()
    widget(app.selectbox, "形状の母集団").set_value(dist).run()
    healthy(app)
    np.testing.assert_array_equal(original(app), panel(dist)["samples"][0])
    assert widget(app.selectbox, "形状ビューのmaster seed").value == 0
    assert widget(app.selectbox, "形状ビューの原標本ID").value == 0
    assert widget(app.selectbox, "形状ビューの内側系列 k").value == 0
    curve = plots(app)[0]["data"][0]
    x, y = array(curve["x"]), array(curve["y"])
    distribution = distribution_shape.population_distribution(dist)
    np.testing.assert_allclose(y, distribution.pdf(x), rtol=1e-14)
    assert np.isfinite(x).all() and np.isfinite(y).all() and (y > 0).all()
    np.testing.assert_allclose([x[0], x[-1]], distribution.ppf([0.001, 0.999]))
    assert plots(app)[0]["layout"]["yaxis"]["type"] == "log"
    if dist == "gamma_0.5":
        bound = -np.sqrt(0.5)
        assert np.isinf(distribution.pdf(bound))  # true mathematical singularity retained
        assert (x > bound).all()  # endpoint omitted from finite display, no replacement
        assert widget(app.metric, "理論超過尖度").value == "12"
        assert any("無限大をゼロや巨大な有限値へ置き換えたりしていません" in item.value for item in app.caption)


def test_B_k_and_method_changes_preserve_original_and_use_exact_endpoints():
    app = open_view()
    before = original(app).copy()
    widget(app.selectbox, "形状ビューのbootstrap回数 B").set_value(999).run()
    widget(app.selectbox, "形状ビューの内側系列 k").set_value(7).run()
    widget(app.multiselect, "形状ビューで表示する手法").set_value(["bca"]).run()
    healthy(app)
    np.testing.assert_array_equal(original(app), before)
    saved = panel()
    trace = next(item for item in plots(app)[1]["data"] if item["name"] == "BCa · 選択k=7")
    np.testing.assert_array_equal(array(trace["x"]), [saved["lower"][0, 0, 2, 7], saved["upper"][0, 0, 2, 7]])
    assert trace["marker"]["line"]["width"] >= 2
    diagnostics = app.dataframe[0].value.iloc[0]
    for key in ("z0", "a", "q_low", "q_high", "tau_low", "tau_high"):
        assert diagnostics[key] == saved[key][0, 0, 2, 7]
    widget(app.multiselect, "形状ビューで表示する手法").set_value([]).run()
    healthy(app)
    np.testing.assert_array_equal(original(app), before)
    assert len(plots(app)[1]["data"]) == 1


def test_each_population_restores_its_own_seed_outer_k_and_hash():
    app = open_view()
    widget(app.selectbox, "形状ビューのmaster seed").set_value(2).run()
    widget(app.selectbox, "形状ビューの原標本ID").set_value(11).run()
    widget(app.selectbox, "形状ビューの内側系列 k").set_value(8).run()
    normal = original(app).copy()
    widget(app.selectbox, "形状の母集団").set_value("gamma_0.5").run()
    assert not np.array_equal(original(app), normal)
    assert widget(app.selectbox, "形状ビューのmaster seed").value == 0
    assert widget(app.selectbox, "形状ビューの原標本ID").value == 0
    widget(app.selectbox, "形状ビューの原標本ID").set_value(399).run()
    widget(app.selectbox, "形状ビューの内側系列 k").set_value(4).run()
    gamma = original(app).copy()
    widget(app.selectbox, "形状の母集団").set_value("normal").run()
    assert widget(app.selectbox, "形状ビューのmaster seed").value == 2
    assert widget(app.selectbox, "形状ビューの原標本ID").value == 11
    assert widget(app.selectbox, "形状ビューの内側系列 k").value == 8
    np.testing.assert_array_equal(original(app), normal)
    widget(app.selectbox, "形状の母集団").set_value("gamma_0.5").run()
    assert widget(app.selectbox, "形状ビューの原標本ID").value == 399
    assert widget(app.selectbox, "形状ビューの内側系列 k").value == 4
    np.testing.assert_array_equal(original(app), gamma)


def test_mode_and_lab_roundtrips_preserve_distribution_ids_and_original():
    app = open_view()
    widget(app.selectbox, "形状の母集団").set_value("gamma_16").run()
    widget(app.selectbox, "形状ビューの原標本ID").set_value(6).run()
    widget(app.selectbox, "形状ビューの内側系列 k").set_value(3).run()
    widget(app.selectbox, "形状ビューのbootstrap回数 B").set_value(999).run()
    saved = original(app).copy()
    widget(app.radio, "推論モード").set_value("再標本化回数と端点の安定性").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    widget(app.radio, "Labを選択").set_value("Chaos Explorer").run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    healthy(app)
    assert widget(app.radio, "推論モード").value == MODE
    assert widget(app.selectbox, "形状の母集団").value == "gamma_16"
    assert widget(app.selectbox, "形状ビューの原標本ID").value == 6
    assert widget(app.selectbox, "形状ビューの内側系列 k").value == 3
    assert widget(app.selectbox, "形状ビューのbootstrap回数 B").value == 999
    np.testing.assert_array_equal(original(app), saved)


def test_pooled_all36_table_and_downloads_are_exact_saved_sources(monkeypatch):
    captured = {}
    actual_download = shape_ui.st.download_button

    def capture(label, data, **kwargs):
        captured[kwargs["file_name"]] = data
        return actual_download(label, data, **kwargs)

    monkeypatch.setattr(shape_ui.st, "download_button", capture)
    app = open_view()
    expected = pd.read_csv(shape_ui.EVIDENCE / "pooled_summary.csv")
    pd.testing.assert_frame_equal(app.dataframe[-1].value, expected, check_dtype=False)
    assert len(app.dataframe[-1].value) == 36
    assert set(app.dataframe[2].value.original_total) == {1200}
    assert captured["rq_m04_pooled_summary.csv"] == (shape_ui.EVIDENCE / "pooled_summary.csv").read_bytes()
    assert captured["rq_m04_overview.html"] == (shape_ui.EVIDENCE / "figures" / "overview.html").read_bytes()


def test_all_six_population_coverage_99MC_intervals_and_direction_signs_match_CSV():
    app = open_view()
    saved = pd.read_csv(shape_ui.EVIDENCE / "pooled_summary.csv")
    for trace in plots(app)[2]["data"]:
        method = next(key for key, label in shape_ui.METHODS.items() if label == trace["name"])
        expected = saved[(saved.B == 4999) & (saved.method == method)].set_index("distribution").loc[list(shape_ui.LABELS)]
        np.testing.assert_allclose(array(trace["y"]), expected.C, rtol=1e-14)
        np.testing.assert_allclose(array(trace["error_y"]["array"]), expected.C_decision_high - expected.C, rtol=1e-13)
        np.testing.assert_allclose(array(trace["error_y"]["arrayminus"]), expected.C - expected.C_decision_low, rtol=1e-13)
    expected = saved[(saved.B == 4999) & (saved.method == "bca")].set_index("distribution").loc[list(shape_ui.LABELS)]
    for trace, metric in zip(plots(app)[3]["data"], ["Mbelow", "Mabove", "T"]):
        np.testing.assert_allclose(array(trace["y"]), expected[metric], rtol=1e-14)
    np.testing.assert_allclose(expected["T"], expected.Mbelow - expected.Mabove, atol=1e-15)
    assert any("95%信頼区間は異なる区間" in item.value for item in app.markdown)


def test_raw_endpoint_sd_S_Ase_Apop_and_local_vs_pooled_values_match_saved_CSV():
    app = open_view()
    originals = pd.read_csv(shape_ui.EVIDENCE / "original_metrics.csv")
    expected = originals[(originals.distribution == "normal") & (originals.master_seed == 0) & (originals.outer_id == 0)]
    for metric in ("se", "lower_sd", "upper_sd", "S", "Ase", "Apop", "C", "T", "mean_width", "warning_rate"):
        np.testing.assert_allclose(app.dataframe[1].value[metric], expected[metric], rtol=1e-13)
    pooled = pd.read_csv(shape_ui.EVIDENCE / "pooled_summary.csv")
    expected = pooled[(pooled.B == 4999) & (pooled.method == "bca")].set_index("distribution").loc[list(shape_ui.LABELS)]
    for trace, metric in zip(plots(app)[4]["data"], ("Ase", "Apop")):
        assert trace["name"] == metric
        np.testing.assert_allclose(array(trace["y"]), expected[metric], rtol=1e-14)


def test_selected_population_B_BCa_metrics_read_exact_pooled_CSV_values():
    app = open_view()
    widget(app.selectbox, "形状の母集団").set_value("gamma_0.5").run()
    widget(app.selectbox, "形状ビューのbootstrap回数 B").set_value(999).run()
    healthy(app)
    saved = pd.read_csv(shape_ui.EVIDENCE / "pooled_summary.csv")
    row = saved[(saved.distribution == "gamma_0.5") & (saved.B == 999) & (saved.method == "bca")].iloc[0]
    for label, key, fmt in [("BCa被覆率", "C", ".2%"), ("被覆MCSE", "C_mcse", ".2%"),
                            ("方向差 T", "T", ".2%"), ("Ase", "Ase", ".4g"), ("Apop", "Apop", ".4g")]:
        assert widget(app.metric, label).value == format(row[key], fmt)
    assert any("B=999 ／ BCa ／ 独立1,200原標本" in item.value for item in app.caption)


def test_old_free_alpha_does_not_change_fixed_alpha_and_both_hypotheses_remain_hold():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.selectbox, "有意水準 α").set_value(0.10).run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    healthy(app)
    assert "固定α=0.05" in app.success[0].value
    text = " ".join(item.value for item in app.markdown)
    assert "H1：保留" in text and "H2：保留" in text
    assert any("歪度単独の因果効果" in item.value for item in app.info)
    assert any("B4999を真値・無限Bの正解とせず" in item.value for item in app.info)
    widget(app.radio, "推論モード").set_value("t推論の探索").run()
    assert widget(app.selectbox, "有意水準 α").value == 0.10


def test_corrupted_original_hash_stops_sample_display(monkeypatch):
    saved = panel()
    saved["samples"][0, 0] += 1
    monkeypatch.setattr(shape_ui, "_panel", lambda *args: saved)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    assert not app.exception and len(app.error) == 1
    assert not app.get("plotly_chart")
