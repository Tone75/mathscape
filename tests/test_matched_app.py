"""Saved M05 viewer: exact evidence values and state, no inference or draws."""

import base64
import copy
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
                       matched_moments, matched_ui, stability)
from scripts import run_matched_benchmark


APP = Path(__file__).resolve().parents[1] / "app.py"
MODE = "同じ歪度、違う分布"


@pytest.fixture(autouse=True)
def forbid_inference_generation_and_duplicate_defaults(monkeypatch, caplog):
    def forbidden(*args, **kwargs):
        raise AssertionError("saved M05 view must never infer, generate or run an experiment")

    for module, names in [(matched_moments, ["rng_for_panel", "generate_sample", "generate_bootstrap_means", "bootstrap_means", "prepare_sample", "infer_single", "original_metrics"]),
                          (distribution_shape, ["rng_for_panel", "generate_sample", "generate_bootstrap_means", "bootstrap_means"]),
                          (stability, ["rng_for_panel", "generate_sample", "generate_bootstrap_means", "bootstrap_means", "prepare_sample", "infer_prefixes"]),
                          (bootstrap, ["draw_replicate", "bootstrap_means", "intervals"]), (bootstrap_ui, ["draw_replicate", "intervals"]),
                          (inference, ["generate_samples", "t_inference"]), (inference_ui, ["generate_samples", "t_inference"]),
                          (run_matched_benchmark, ["run_benchmark", "run_smoke", "run_cell", "generate_sample", "generate_bootstrap_means"])]:
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
    return array(plots(app)[1]["data"][0]["x"])


def panel(dist="gamma", seed=0):
    with np.load(matched_ui.EVIDENCE / "panels" / f"{dist}_seed{seed}.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def test_initial_saved_example_fixed_metadata_and_K1_stability_NA_for_all_methods():
    app = open_view()
    saved = panel()
    np.testing.assert_array_equal(original(app), saved["samples"][0])
    digest = hashlib.sha256(original(app).astype("<f8").tobytes()).hexdigest()
    assert digest == saved["sample_sha256"][0].decode("ascii")
    assert any(digest in item.value for item in app.caption)
    assert "固定α=.05・B4999・K1" in app.success[0].value
    assert app.dataframe[1].value[matched_ui.STABILITY].isna().all().all()
    assert any("全方法（tも含む）でNA／対象外" in item.value for item in app.info)
    assert not any(item.label in ("Ase", "Apop", "端点SD") for item in app.metric)
    assert plots(app)[1]["data"][0]["marker"]["symbol"] == "circle"


def test_two_PDFs_share_exact_X_axis_and_tail_omission_probabilities_without_renormalizing():
    app = open_view()
    figure, facts = plots(app)[0], app.dataframe[0].value
    first, second = (array(trace["x"]) for trace in figure["data"])
    np.testing.assert_array_equal(first, second)
    for trace, dist in zip(figure["data"], matched_ui.LABELS):
        population = matched_moments.population_distribution(dist)
        np.testing.assert_allclose(array(trace["y"]), population.pdf(first), rtol=1e-14)
        row = facts.loc[facts["母集団"] == matched_ui.LABELS[dist]].iloc[0]
        assert row["母平均"] == 0 and row["母分散"] == 1 and row["母歪度"] == 2
        assert row["表示窓より下の確率"] == population.cdf(first[0])
        assert row["表示窓より上の確率"] == population.sf(first[-1])
    assert facts.iloc[0]["超過尖度"] != facts.iloc[1]["超過尖度"]
    assert facts.iloc[0]["支持下限"] != facts.iloc[1]["支持下限"]
    assert any("再正規化していません" in item.value for item in app.caption)


@pytest.mark.parametrize("dist", ["gamma", "lognormal"])
def test_original_and_all_three_saved_95percent_intervals_exact(dist):
    app = open_view()
    widget(app.selectbox, "一致モーメントの母集団").set_value(dist).run()
    healthy(app)
    saved = panel(dist)
    np.testing.assert_array_equal(original(app), saved["samples"][0])
    for trace, mi in zip(plots(app)[1]["data"][1:], range(3)):
        np.testing.assert_array_equal(array(trace["x"]), [saved["lower"][0, mi], saved["upper"][0, mi]])
        assert trace["marker"]["line"]["width"] >= 2
    for mi, row in app.dataframe[1].value.iterrows():
        for key in ("z0", "a", "q_low", "q_high", "tau_low", "tau_high"):
            assert row[key] == pytest.approx(saved[key][0, mi], nan_ok=True)


def test_per_population_ID_restoration_and_method_mode_lab_hash_preservation():
    app = open_view()
    widget(app.selectbox, "一致モーメントのmaster seed").set_value(2).run()
    widget(app.number_input, "一致モーメントの原標本ID").set_value(19).run()
    gamma = original(app).copy()
    widget(app.selectbox, "一致モーメントの母集団").set_value("lognormal").run()
    assert widget(app.selectbox, "一致モーメントのmaster seed").value == 0
    assert widget(app.number_input, "一致モーメントの原標本ID").value == 0
    assert not np.array_equal(original(app), gamma)
    widget(app.number_input, "一致モーメントの原標本ID").set_value(9999).run()
    lognormal = original(app).copy()
    widget(app.multiselect, "一致モーメントで表示する手法").set_value(["bca"]).run()
    widget(app.radio, "推論モード").set_value("分布の形と被覆").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    widget(app.radio, "Labを選択").set_value("Fractal Explorer").run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    healthy(app)
    assert widget(app.radio, "推論モード").value == MODE
    assert widget(app.number_input, "一致モーメントの原標本ID").value == 9999
    assert widget(app.multiselect, "一致モーメントで表示する手法").value == ["bca"]
    np.testing.assert_array_equal(original(app), lognormal)
    widget(app.selectbox, "一致モーメントの母集団").set_value("gamma").run()
    assert widget(app.selectbox, "一致モーメントのmaster seed").value == 2
    assert widget(app.number_input, "一致モーメントの原標本ID").value == 19
    np.testing.assert_array_equal(original(app), gamma)
    widget(app.multiselect, "一致モーメントで表示する手法").set_value([]).run()
    healthy(app)
    np.testing.assert_array_equal(original(app), gamma)
    assert len(plots(app)[1]["data"]) == 1


def test_all_six_pooled_rows_and_coverage_error_bars_match_saved_CSV():
    app = open_view()
    expected = pd.read_csv(matched_ui.EVIDENCE / "pooled_summary.csv")
    actual = app.dataframe[-2].value
    pd.testing.assert_frame_equal(actual.astype(object).where(actual.notna(), None),
                                  expected.astype(object).where(expected.notna(), None), check_dtype=False)
    assert len(app.dataframe[-2].value) == 6 and len(app.dataframe[-1].value) == 18
    assert set(app.dataframe[2].value.original_total) == {30000}
    for trace, dist in zip(plots(app)[2]["data"], matched_ui.LABELS):
        rows = expected[expected.distribution == dist].set_index("method").loc[list(matched_ui.METHODS)]
        np.testing.assert_allclose(array(trace["y"]), rows.C, rtol=1e-14)
        np.testing.assert_allclose(array(trace["error_y"]["array"]), rows.C_decision_high - rows.C, rtol=1e-13)
        np.testing.assert_allclose(array(trace["error_y"]["arrayminus"]), rows.C - rows.C_decision_low, rtol=1e-13)
    assert app.dataframe[-2].value[matched_ui.STABILITY].isna().all().all()


def test_primary_difference_plot_strict_band_and_equivalence_scope_read_saved_hypotheses():
    app = open_view()
    saved = json.loads((matched_ui.EVIDENCE / "hypotheses.json").read_text())
    figure = plots(app)[3]
    for trace, name in zip(figure["data"], ("H1", "H2")):
        row = saved["quantities"][name]
        assert trace["x"] == [row["difference"]]
        assert trace["error_x"]["array"] == [row["high"] - row["difference"]]
        assert trace["error_x"]["arrayminus"] == [row["difference"] - row["low"]]
        assert row["status"] == "not_supported" and row["equivalence_supported"]
    band = next(shape for shape in figure["layout"]["shapes"] if shape["type"] == "rect")
    assert band["x0"] == -0.01 and band["x1"] == 0.01
    assert any("両指標で±1pp内" in item.value and "名目95%を満たすという意味ではありません" in item.value for item in app.success)
    assert widget(app.metric, "BCa被覆 G").value == "91.377%"
    assert widget(app.metric, "BCa被覆 L").value == "91.173%"
    assert widget(app.metric, "BCa被覆差 ΔC（L−G）").value == "-0.203 pp"
    assert widget(app.metric, "BCa方向差 ΔT（L−G）").value == "+0.197 pp"
    assert any("1pp=.01の絶対的な確率差" in item.value for item in app.caption)
    assert any("全seed同符号という追加ゲートはありません" in item.value for item in app.caption)


def test_downloaded_summary_and_HTML_are_exact_and_sample_CSV_reconstructs_original_hash(monkeypatch):
    captured = {}
    actual = matched_ui.st.download_button

    def capture(label, data, **kwargs):
        captured[kwargs["file_name"]] = data
        return actual(label, data, **kwargs)

    monkeypatch.setattr(matched_ui.st, "download_button", capture)
    app = open_view()
    assert captured["rq_m05_pooled_summary.csv"] == (matched_ui.EVIDENCE / "pooled_summary.csv").read_bytes()
    assert captured["rq_m05_overview.html"] == (matched_ui.EVIDENCE / "figures" / "overview.html").read_bytes()
    rows = list(csv.DictReader(io.StringIO(captured["rq_m05_gamma_seed0_outer0.csv"].decode())))
    assert len(rows) == 20
    values = np.array([float(row["X"]) for row in rows], dtype="<f8")
    np.testing.assert_array_equal(values, original(app))
    assert hashlib.sha256(values.tobytes()).hexdigest() == rows[0]["sample_sha256"]
    assert [int(row["observation_index"]) for row in rows] == list(range(20))


def test_invalid_descriptive_main_values_do_not_silently_use_conditional_values(monkeypatch):
    actual_csv = matched_ui._csv

    def modified(path, modified_time):
        rows, payload = actual_csv(path, modified_time)
        if Path(path).name == "pooled_summary.csv":
            rows = copy.deepcopy(rows)
            row = next(row for row in rows if row["distribution"] == "gamma" and row["method"] == "t")
            row.update(C="nan", C_mcse="nan", C_decision_low="nan", C_decision_high="nan", invalid_count="1", conditional_C="0.123")
        return rows, payload

    monkeypatch.setattr(matched_ui, "_csv", modified)
    app = open_view()
    row = app.dataframe[2].value.loc[(app.dataframe[2].value.distribution == "gamma") & (app.dataframe[2].value.method == "t")].iloc[0]
    assert pd.isna(row.C)
    assert any("conditional_の参考値へ置換せず" in item.value for item in app.warning)
    assert widget(app.metric, "BCa被覆 G").value == "91.377%"  # unrelated t invalid does not poison BCa


def test_free_alpha_is_preserved_separately_from_fixed_alpha_B_K():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.selectbox, "有意水準 α").set_value(0.10).run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    healthy(app)
    assert "固定α=.05・B4999・K1" in app.success[0].value
    assert not any("有意水準" in item.label for item in app.selectbox)
    widget(app.radio, "推論モード").set_value("t推論の探索").run()
    assert widget(app.selectbox, "有意水準 α").value == 0.10


def test_original_byte_hash_mismatch_prevents_rendering(monkeypatch):
    saved = panel()
    saved["samples"][0, 0] += 1
    monkeypatch.setattr(matched_ui, "_panel", lambda *args: saved)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app.radio, "Labを選択").set_value("Inference Lab").run()
    widget(app.radio, "推論モード").set_value(MODE).run()
    assert not app.exception and len(app.error) == 1
    assert not app.get("plotly_chart")
