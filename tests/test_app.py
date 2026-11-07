"""Exercise actual Streamlit reruns; browser layout is verified separately."""

import base64
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.elements.lib import policies

from mathscape import probability


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def reject_duplicate_widget_defaults(monkeypatch, caplog):
    # Streamlit logs this warning only once per process. Re-arm it per test
    # so an earlier test cannot hide a state/default conflict in a later one.
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    caplog.set_level(logging.WARNING, logger=policies.__name__)
    yield
    assert not any(
        "was created with a default value" in record.getMessage()
        and "Session State API" in record.getMessage()
        for record in caplog.get_records("call")
    ), "A widget duplicates its Session State value with an explicit default"


def new_app():
    return AppTest.from_file(str(APP), default_timeout=20).run()


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def assert_healthy(app):
    assert not app.exception
    assert not app.error


def plots(app):
    return [json.loads(element.proto.spec) for element in app.get("plotly_chart")]


def array(values):
    """Decode either ordinary Plotly arrays or Plotly's typed-array transport."""
    if isinstance(values, dict) and "bdata" in values:
        result = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        if "shape" in values:
            result = result.reshape(tuple(int(size) for size in values["shape"].split(",")))
        return result
    return np.asarray(values)


def test_initial_normal_distribution_and_parameter_rerun():
    app = new_app()
    assert_healthy(app)
    assert app.title[0].value == "MATHSCAPE"
    assert widget(app.slider, "平均 μ").value == 0.0
    assert widget(app.slider, "標準偏差 σ").value == 1.0
    assert widget(app.metric, "平均 E[X]").value == "0"
    assert widget(app.metric, "分散 Var[X]").value == "1"
    assert r"\sigma" in app.latex[0].value
    assert "有限区間" in app.info[0].value
    before = array(plots(app)[0]["data"][0]["y"]).copy()
    widget(app.slider, "平均 μ").set_value(1.5)
    widget(app.slider, "標準偏差 σ").set_value(2.0)
    app.run()
    assert_healthy(app)
    assert widget(app.metric, "平均 E[X]").value == "1.5"
    assert widget(app.metric, "分散 Var[X]").value == "4"
    after = plots(app)[0]["data"][0]
    assert not np.array_equal(before, array(after["y"]))
    np.testing.assert_allclose(array(after["y"]), probability.normal_pdf(array(after["x"]), 1.5, 2.0))


@pytest.mark.parametrize("distribution, sliders, formula, mean, variance", [
    ("ガンマ分布", {"shape k": 0.5, "scale θ": 2.0}, r"\Gamma", 1.0, 2.0),
    ("ベータ分布", {"形状 α": 0.5, "形状 β": 0.5}, r"B(\alpha,\beta)", 0.5, 0.125),
])
def test_singular_distribution_endpoints_are_excluded_and_explained(
    distribution, sliders, formula, mean, variance
):
    app = new_app()
    widget(app.selectbox, "分布").select(distribution).run()
    for label, value in sliders.items():
        widget(app.slider, label).set_value(value)
    app.run()
    assert_healthy(app)
    assert formula in app.latex[0].value
    assert "無限大" in app.info[0].value
    assert "除外" in app.info[0].value
    assert float(widget(app.metric, "平均 E[X]").value) == pytest.approx(mean)
    assert float(widget(app.metric, "分散 Var[X]").value) == pytest.approx(variance)
    trace = plots(app)[0]["data"][0]
    x, y = array(trace["x"]), array(trace["y"])
    assert np.isfinite(y).all()
    assert (y >= 0).all()
    assert (x > 0).all()
    if distribution == "ベータ分布":
        assert (x < 1).all()
        np.testing.assert_allclose(y, probability.beta_pdf(x, 0.5, 0.5))
    else:
        np.testing.assert_allclose(y, probability.gamma_pdf(x, 0.5, 2.0))


def test_lab_navigation_formulas_and_explanations():
    app = new_app()
    for lab, formula, explanation in [
        ("Chaos Explorer", r"x_{n+1}", "決定論的モデル"),
        ("Fractal Explorer", r"z_{n+1}", "非発散観測点"),
        ("Probability Explorer", r"f(x)", "定義域"),
    ]:
        widget(app.radio, "Labを選択").set_value(lab).run()
        assert_healthy(app)
        assert app.header[0].value == lab
        assert formula in app.latex[0].value
        text = " ".join(element.value for element in app.markdown) + " ".join(element.value for element in app.info)
        assert explanation in text
        assert len(plots(app)) >= 1


def test_chaos_preset_boundary_initial_value_and_plotted_difference():
    app = new_app()
    widget(app.radio, "Labを選択").set_value("Chaos Explorer").run()
    widget(app.selectbox, "観察プリセット").select("周期2 · r=3.2").run()
    assert widget(app.slider, "成長パラメータ r").value == 3.2
    widget(app.slider, "初期値 x₀").set_value(1.0)
    widget(app.slider, "反復回数 N").set_value(120)
    widget(app.slider, "初期差の指数 log₁₀(δ)").set_value(-12)
    app.run()
    assert_healthy(app)
    specs = plots(app)
    assert len(specs) == 3
    first = array(specs[0]["data"][0]["y"])
    second = array(specs[0]["data"][1]["y"])
    difference = array(specs[1]["data"][0]["y"])
    assert first.shape == second.shape == difference.shape == (121,)
    assert first[0] == 1.0
    assert 0 <= second[0] < 1.0
    assert first[1] == 0.0
    np.testing.assert_array_equal(difference, np.abs(first - second))
    assert array(specs[2]["data"][0]["x"]).size == 400 * 80
    assert array(specs[2]["data"][0]["y"]).size == 400 * 80


def test_fractal_zoom_pan_reset_updates_calculated_domain():
    app = new_app()
    widget(app.radio, "Labを選択").set_value("Fractal Explorer").run()
    widget(app.button, "＋ ズームイン").click().run()
    assert widget(app.number_input, "表示幅（実軸・虚軸共通）").value == 1.5
    widget(app.button, "←").click().run()
    assert widget(app.number_input, "中心 Re(c)").value == pytest.approx(-1.075)
    widget(app.button, "↑").click().run()
    assert widget(app.number_input, "中心 Im(c)").value == pytest.approx(0.375)
    trace = plots(app)[0]["data"][0]
    x, y = array(trace["x"]), array(trace["y"])
    assert (x[0] + x[-1]) / 2 == pytest.approx(-1.075)
    assert (y[0] + y[-1]) / 2 == pytest.approx(0.375)
    assert x[-1] - x[0] == pytest.approx(1.5)
    widget(app.button, "↓").click().run()
    widget(app.button, "→").click().run()
    assert widget(app.number_input, "中心 Re(c)").value == pytest.approx(-0.7)
    assert widget(app.number_input, "中心 Im(c)").value == pytest.approx(0.0)
    widget(app.button, "− ズームアウト").click().run()
    assert widget(app.number_input, "表示幅（実軸・虚軸共通）").value == 3.0
    widget(app.number_input, "中心 Re(c)").set_value(-0.75)
    widget(app.number_input, "中心 Im(c)").set_value(0.1)
    app.run()
    widget(app.button, "全体表示に戻す").click().run()
    assert widget(app.number_input, "中心 Re(c)").value == -0.7
    assert widget(app.number_input, "中心 Im(c)").value == 0.0
    assert widget(app.number_input, "表示幅（実軸・虚軸共通）").value == 3.0
    assert_healthy(app)


def test_fractal_maximum_controls_and_unconfirmed_color_layer():
    app = new_app()
    widget(app.radio, "Labを選択").set_value("Fractal Explorer").run()
    widget(app.slider, "解像度（一辺の画素数）").set_value(450)
    widget(app.slider, "最大反復回数").set_value(150)
    app.run()
    assert_healthy(app)
    spec = plots(app)[0]
    escaped = array(spec["data"][0]["z"])
    unconfirmed = array(spec["data"][1]["z"])
    assert escaped.shape == unconfirmed.shape == (450, 450)
    assert np.isfinite(escaped).any()
    assert np.isfinite(unconfirmed).any()
    assert np.array_equal(np.isfinite(escaped), ~np.isfinite(unconfirmed))
    assert np.nanmin(escaped) >= 1
    assert np.nanmax(escaped) <= 150
    assert spec["layout"]["yaxis"]["scaleanchor"] == "x"
    assert spec["layout"]["yaxis"]["scaleratio"] == 1


def test_nonfinite_density_is_reported_without_rendering_invalid_plot(monkeypatch):
    monkeypatch.setattr(
        probability, "probability_curve",
        lambda *args, **kwargs: (np.array([0.0, 1.0]), np.array([1.0, np.inf])),
    )
    app = new_app()
    assert not app.exception
    assert len(app.error) == 1
    assert "非有限" in app.error[0].value
    assert not plots(app)


@pytest.mark.parametrize("lab, key, label, initial, edited", [
    ("Fractal Explorer", "f_real", "中心 Re(c)", -0.7, -0.75),
    ("Fractal Explorer", "f_imag", "中心 Im(c)", 0.0, 0.1),
    ("Fractal Explorer", "f_width", "表示幅（実軸・虚軸共通）", 3.0, 0.8),
    ("Chaos Explorer", "chaos_r", "成長パラメータ r", 3.8, 3.6),
])
def test_state_initialization_manual_edit_and_lab_reentry(lab, key, label, initial, edited):
    app = new_app()
    widget(app.radio, "Labを選択").set_value(lab).run()
    elements = app.slider if key == "chaos_r" else app.number_input
    assert widget(elements, label).value == initial
    assert app.session_state[key] == initial

    # A pre-existing state value must survive a rerun without a widget default
    # competing with it. Then verify direct user editing and another rerun.
    app.session_state[key] = edited
    app.run()
    elements = app.slider if key == "chaos_r" else app.number_input
    assert widget(elements, label).value == edited
    widget(elements, label).set_value(initial).run()
    app.run()
    assert app.session_state[key] == initial
    assert_healthy(app)

    # The existing app resets hidden widgets on leaving a Lab. Preserve that
    # behavior and ensure defaults are correctly initialized on returning.
    elements = app.slider if key == "chaos_r" else app.number_input
    widget(elements, label).set_value(edited).run()
    widget(app.radio, "Labを選択").set_value("Probability Explorer").run()
    widget(app.radio, "Labを選択").set_value(lab).run()
    assert app.session_state[key] == initial
    assert_healthy(app)


@pytest.mark.parametrize("preset, expected", [
    ("安定固定点 · r=2.5", 2.5),
    ("周期2 · r=3.2", 3.2),
    ("複雑な軌道 · r=3.9", 3.9),
])
def test_chaos_presets_and_manual_slider_remain_independent(preset, expected):
    app = new_app()
    widget(app.radio, "Labを選択").set_value("Chaos Explorer").run()
    widget(app.selectbox, "観察プリセット").select(preset).run()
    assert app.session_state.chaos_r == expected
    widget(app.slider, "成長パラメータ r").set_value(3.6).run()
    app.run()
    assert app.session_state.chaos_r == 3.6
    widget(app.selectbox, "観察プリセット").select("自由探索").run()
    assert app.session_state.chaos_r == 3.6
    widget(app.selectbox, "観察プリセット").select(preset).run()
    assert app.session_state.chaos_r == expected
    assert_healthy(app)
