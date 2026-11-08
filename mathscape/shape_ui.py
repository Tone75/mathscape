"""Read-only RQ-M04 population-shape and saved-interval exploration."""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mathscape.distribution_shape import population_distribution, population_info


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "rq_m04"
STATE_KEY = "shape_saved_view"
LABELS = {"normal": "正規分布（独立した対照）", "gamma_0.5": "標準化ガンマ r=0.5", "gamma_1": "標準化ガンマ r=1",
          "gamma_2": "標準化ガンマ r=2", "gamma_4": "標準化ガンマ r=4", "gamma_16": "標準化ガンマ r=16"}
METHODS = {"t": "t", "percentile": "percentile", "bca": "BCa"}
COLORS = {"t": "#46DCC0", "percentile": "#FFB86B", "bca": "#A8A2FF"}


def _state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = dict(distribution="normal", B=4999, methods=list(METHODS),
                                         selections={dist: dict(seed=0, outer=0, k=0) for dist in LABELS})
    return st.session_state[STATE_KEY]


def _switch_population():
    state = _state()
    state["distribution"] = st.session_state.shape_distribution
    for key, value in state["selections"][state["distribution"]].items():
        st.session_state[f"shape_{key}"] = value


def _remember():
    state = _state()
    selection = state["selections"][state["distribution"]]
    for key in ("seed", "outer", "k"):
        if f"shape_{key}" in st.session_state:
            selection[key] = st.session_state[f"shape_{key}"]
    for key in ("B", "methods"):
        if f"shape_{key}" in st.session_state:
            state[key] = st.session_state[f"shape_{key}"]


@st.cache_data(max_entries=4, show_spinner=False)
def _panel(path, modified):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


@st.cache_data(max_entries=4, show_spinner=False)
def _csv(path, modified):
    payload = Path(path).read_bytes()
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8")))), payload


@st.cache_data(max_entries=12, show_spinner=False)
def _original_rows(path, modified, distribution, seed, outer):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["distribution"] == distribution
                and int(row["master_seed"]) == seed and int(row["outer_id"]) == outer]


def _display(rows):
    converted = []
    for row in rows:
        result = {}
        for key, value in row.items():
            try:
                number = float(value)
                result[key] = number if np.isfinite(number) else None
            except (ValueError, TypeError):
                result[key] = value
        converted.append(result)
    return converted


def _chart(figure, x, y, height=370):
    figure.update_layout(template="plotly_dark", height=height,
                         xaxis_title=x, yaxis_title=y, margin=dict(l=40, r=25, t=70, b=40),
                         legend=dict(orientation="h", y=1.2), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _population(dist):
    info, distribution = population_info(dist), population_distribution(dist)
    cols = st.columns(3)
    cols[0].metric("理論歪度", f"{info['skewness']:.5g}")
    cols[1].metric("理論超過尖度", f"{info['excess_kurtosis']:.5g}")
    cols[2].metric("理論支持域の下限", "−∞" if not np.isfinite(info["lower_bound"]) else f"{info['lower_bound']:.5g}")
    if dist == "normal":
        st.latex(r"X\sim N(0,1)")
    else:
        st.latex(r"G\sim\mathrm{Gamma}(\mathrm{shape}=r,\mathrm{scale}=1),\quad X=(G-r)/\sqrt r")
        st.caption(f"r={info['shape']:g}。理論母平均・母分散を使った標準化で、各原標本を平均0・分散1へ再標準化していません。")
    # Quantile coordinates omit both tails and never include a singular bound.
    x = distribution.ppf(np.linspace(0.001, 0.999, 700))
    y = distribution.pdf(x)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and (y > 0).all()):
        raise ValueError("finite population display grid unavailable")
    figure = go.Figure(go.Scatter(x=x, y=y, name="真の母集団PDF", line=dict(color="#46DCC0", width=3)))
    figure.update_yaxes(type="log")
    _chart(figure, "X（標準化単位）", "確率密度 f(x)（対数目盛）", 300)
    st.caption("PDFの表示窓は0.1%〜99.9%分位点で、両端の裾と支持域端点を省略しています。"
               "縦軸は対数目盛で、この対数表示の見た目の面積を確率とは読めません。表示密度を切り詰めたり、無限大をゼロや巨大な有限値へ置き換えたりしていません。"
               "r=0.5では下限へ近づくと密度が無限大になります。表示曲線は有限格子の近似です。")
    st.info("rとともに歪度・尖度・支持域が連動する、ガンマ族内の分布形状の比較です。歪度単独の因果効果ではありません。"
            "正規対照はr=16と同一ではなく、r=1は分布としてExp(1)−1に一致しますが、今回の新しい独立パネルです。")


def _sample(panel, state, selection):
    oi, bi, k = selection["outer"], list(panel["B_values"]).index(state["B"]), selection["k"]
    sample = panel["samples"][oi]
    fig = go.Figure(go.Scatter(x=sample, y=np.full(sample.size, -4), mode="markers", name="保存原標本の観測点",
                              marker=dict(symbol="circle", size=8, color="white", line=dict(color="#080C14", width=1))))
    diagnostics = []
    for mi, method in enumerate(panel["methods"]):
        if method not in state["methods"]:
            continue
        lower, upper, valid = (panel[key][oi, bi, mi] for key in ("lower", "upper", "valid"))
        if method == "t":
            if valid[0]:
                fig.add_scatter(x=[lower[0], upper[0]], y=[-2, -2], mode="lines+markers", name="t参照（1回）", line_color=COLORS[method])
        else:
            offset = -0.13 if method == "percentile" else 0.13
            xs, ys = [], []
            for inner in np.flatnonzero(valid):
                xs.extend([lower[inner], upper[inner], None])
                ys.extend([float(inner) + offset] * 2 + [None])
            fig.add_scatter(x=xs, y=ys, mode="lines", name=f"{METHODS[method]} · K10系列",
                            line=dict(color=COLORS[method], dash="dash" if method == "percentile" else "dot"))
            if valid[k]:
                fig.add_scatter(x=[lower[k], upper[k]], y=[k + offset] * 2, mode="lines+markers", name=f"{METHODS[method]} · 選択k={k}",
                                line=dict(color=COLORS[method], width=5), marker=dict(symbol="line-ns", size=14, line=dict(color=COLORS[method], width=2)))
        row = dict(method=str(method), B=state["B"], k=k)
        row.update({key: panel[key][oi, bi, mi, k].item() for key in ("lower", "upper", "valid", "reason", "z0", "a", "q_low", "q_high", "tau_low", "tau_high", "tail_warning")})
        diagnostics.append(row)
    fig.add_vline(x=0, line_dash="dot", line_color="white", annotation_text="真の母平均 μ=0")
    _chart(fig, "原標本の観測値／95%信頼区間（標準化単位）", "内側系列 k（下部：原標本点とt参照）", 510)
    st.dataframe(diagnostics, hide_index=True, width="stretch")
    invalid = [row for row in diagnostics if not row["valid"]]
    if invalid:
        st.warning("無効な保存区間：" + " ／ ".join(f"{row['method']}: {row['reason']}" for row in invalid))
    st.caption("いずれかの端点でtau=B×min(q,1−q)<10なら尾部解像度警告です。警告は除外せず、無効は別手法に置き換えません。"
               "tは原標本から作る1回の参照です。K10個の区間やB回の再標本化は、新しい独立な観測データではありません。")
    st.caption("95%は同じ区間手続きの反復被覆についての名目水準で、実現した一つの区間に真値が95%の確率で存在するという意味ではありません。")


def _aggregate(pooled, b, selected_distribution):
    st.subheader("3. 分布形状ごとの被覆・非被覆・端点の揺れ")
    st.write("各分布は3seedをプールした1,200の独立原標本です。各原標本内K10を平均してから原標本間で集計します。"
             "被覆推定の99% Monte Carlo区間と、上の各標本から作る95%信頼区間は異なる区間です。")
    current = next(row for row in pooled if row["distribution"] == selected_distribution and int(row["B"]) == b and row["method"] == "bca")
    st.caption(f"選択した母集団：{LABELS[selected_distribution]} ／ B={b} ／ BCa ／ 独立1,200原標本の保存集計。"
               "上の原標本ID・kの1例や理論モーメントとは別の反復集計です。")
    columns = st.columns(5)
    for column, label, key, fmt in zip(columns, ["BCa被覆率", "被覆MCSE", "方向差 T", "Ase", "Apop"],
                                        ["C", "C_mcse", "T", "Ase", "Apop"], [".2%", ".2%", ".2%", ".4g", ".4g"]):
        value = float(current[key])
        column.metric(label, format(value, fmt) if np.isfinite(value) else "未定義")
    figure = go.Figure()
    for method in METHODS:
        rows = [next(row for row in pooled if row["distribution"] == dist and int(row["B"]) == b and row["method"] == method) for dist in LABELS]
        values = np.array([float(row["C"]) for row in rows])
        figure.add_scatter(x=list(LABELS.values()), y=values, mode="markers+lines", name=METHODS[method],
                           line=dict(color=COLORS[method], dash={"t": "solid", "percentile": "dash", "bca": "dot"}[method]),
                           marker_symbol={"t": "circle", "percentile": "square", "bca": "diamond"}[method],
                           error_y=dict(type="data", symmetric=False, array=np.array([float(row["C_decision_high"]) for row in rows]) - values,
                                        arrayminus=values - np.array([float(row["C_decision_low"]) for row in rows])))
    figure.add_hline(y=0.95, line_dash="dot", annotation_text="名目95%")
    _chart(figure, "独立に生成した母集団（カテゴリ、形状間の対応標本ではありません）", "被覆率Cと99% Monte Carlo区間", 420)
    rows = [next(row for row in pooled if row["distribution"] == dist and int(row["B"]) == b and row["method"] == "bca") for dist in LABELS]
    sides = go.Figure()
    for metric, label, color in (("Mbelow", "Mbelow：区間全体が真値より下（U<0）", "#FFB86B"),
                                  ("Mabove", "Mabove：区間全体が真値より上（L>0）", "#46DCC0"),
                                  ("T", "T=Mbelow−Mabove", "#A8A2FF")):
        sides.add_scatter(x=list(LABELS.values()), y=[float(row[metric]) for row in rows], mode="lines+markers", name=label, line_color=color)
    _chart(sides, "母集団（BCa、選択B）", "非被覆の方向と符号付き差T", 330)
    stability = go.Figure()
    for metric in ("Ase", "Apop"):
        stability.add_scatter(x=list(LABELS.values()), y=[float(row[metric]) for row in rows], mode="lines+markers", name=metric)
    _chart(stability, "母集団（BCa、選択B）", "端点の揺れ：AseとApop（異なる尺度）", 330)
    compact = ["distribution", "B", "method", "original_total", "C", "C_mcse", "C_decision_low", "C_decision_high",
               "Mbelow", "Mabove", "T", "T_mcse", "lower_sd", "upper_sd", "S", "Ase", "Apop", "mean_width", "warning_rate", "valid_count", "invalid_count"]
    st.dataframe(_display([{key: row[key] for key in compact} for row in pooled if int(row["B"]) == b]), hide_index=True, width="stretch")
    st.caption(f"集計図・表はB={b}。被覆図は3方法すべて、方向と尺度の図はBCaを表示し、原標本図の手法選択とは独立しています。"
               "線で結んだ中間形状は記述であり、全域の単調関係を検証したものではありません。C+Mbelow+Mabove=1、T=Mbelow−Maboveです。")


def render_shape_view():
    st.header("Inference Lab · 分布の形と被覆")
    st.caption("保存結果専用 · 母平均0・母分散1・n=20・α=0.05固定。旧自由操作のαとは独立しています。"
               "各セルR400原標本、K10内側系列、B999/4999の共通prefix。標本生成や重いライブ実験は起動しません。")
    state = _state()
    selection = state["selections"][state["distribution"]]
    for key, value in dict(distribution=state["distribution"], B=state["B"], methods=state["methods"], **selection).items():
        st.session_state.setdefault(f"shape_{key}", value)
    st.sidebar.selectbox("形状の母集団", list(LABELS), format_func=LABELS.get, key="shape_distribution", on_change=_switch_population)
    for label, key, values in (("形状ビューのmaster seed", "seed", [0, 1, 2]), ("形状ビューの原標本ID", "outer", list(range(400))),
                                ("形状ビューのbootstrap回数 B", "B", [999, 4999]), ("形状ビューの内側系列 k", "k", list(range(10)))):
        st.sidebar.selectbox(label, values, key=f"shape_{key}", on_change=_remember)
    st.sidebar.multiselect("形状ビューで表示する手法", list(METHODS), format_func=METHODS.get, key="shape_methods", on_change=_remember)
    st.sidebar.caption("分布ごとにseed・原標本ID・kを保持します。分布変更は独立した別原標本への切替で、同じIDでも形状間の対応標本ではありません。戻ると選択IDを復元します。")
    selection = state["selections"][state["distribution"]]
    try:
        path = EVIDENCE / "panels" / f"{state['distribution']}_seed{selection['seed']}.npz"
        panel = _panel(str(path), path.stat().st_mtime_ns)
        pooled, csv_bytes = _csv(str(EVIDENCE / "pooled_summary.csv"), (EVIDENCE / "pooled_summary.csv").stat().st_mtime_ns)
        sample_rows = _original_rows(str(EVIDENCE / "original_metrics.csv"), (EVIDENCE / "original_metrics.csv").stat().st_mtime_ns,
                                     state["distribution"], selection["seed"], selection["outer"])
        hypotheses = json.loads((EVIDENCE / "hypotheses.json").read_text())
        sample = panel["samples"][selection["outer"]]
        digest = hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes()).hexdigest()
        if digest != panel["sample_sha256"][selection["outer"]]:
            raise ValueError("saved original hash mismatch")
        _population(state["distribution"])
    except (OSError, KeyError, ValueError, IndexError) as error:
        st.error(f"RQ-M04の保存結果を表示できません：{type(error).__name__}。証拠ファイルを確認してください。")
        return
    st.success(f"表示条件：{LABELS[state['distribution']]} ／ seed={selection['seed']} ／ 原標本ID={selection['outer']} ／ "
               f"B={state['B']} ／ k={selection['k']} ／ n20・固定α=0.05")
    st.caption(f"原標本SHA-256：{digest}")
    st.subheader("1. 母集団と対応する保存原標本・95%区間")
    _sample(panel, state, selection)
    st.subheader("2. この原標本の端点SDと2つの尺度")
    columns = ["B", "method", "se", "lower_sd", "upper_sd", "S", "Ase", "Apop", "C", "Mbelow", "Mabove", "T", "mean_width", "warning_rate", "complete", "valid_inner", "invalid_inner", "total_inner"]
    st.dataframe(_display([{key: row[key] for key in columns} for row in sample_rows]), hide_index=True, width="stretch")
    st.caption("S=√((SD_L²+SD_U²)/2)、各SDはK系列間のddof=1。Ase=S/(s/√20)、Apop=S/(1/√20)。"
               "Apopの母分散はシミュレーション評価専用で、推論区間へ渡していません。s自体も分布形状で変わるため、生SDと両尺度を併記します。"
               "K10のAを旧RQ-M03のK20と同条件比較しません。")
    _aggregate(pooled, state["B"], state["distribution"])
    st.subheader("4. 事前固定した2仮説の判定")
    st.caption("この判定欄の推定値・区間・閾値は確率差の尺度です。0.01は1パーセントポイントを表します。")
    for name in ("H1", "H2"):
        quantity = hypotheses["quantities"][name]
        estimate = quantity["difference"] if name == "H1" else quantity["mean"]
        status = {"supported": "支持", "not_supported": "不支持", "inconclusive": "保留"}[quantity["status"]]
        def number(value):
            return f"{value:.5g}" if value is not None and np.isfinite(value) else "未定義"

        st.write(f"{name}：{status}。BCa・B4999に固定した推定値{number(estimate)}、99%区間[{number(quantity['low'])}, {number(quantity['high'])}]。"
                 f"事前閾値{quantity['threshold']:g}、seed別点推定：" + ", ".join(number(value) for value in quantity["seed_points"]))
    st.info("H1はr0.5−r16の独立2群被覆差で、99%区間の上限<−0.02、全seed差<0が支持条件。"
            "H2はr0.5のTで、99%区間の下限>0.03、全seed平均T>0が支持条件です。今回の両仮説は保留で、"
            "区間が閾値をまたぐ不確実性を保持します。保留は差ゼロ・左右対称性・同等性の証明ではありません。")
    with st.expander("保存済みプール集計36行をすべて見る"):
        st.dataframe(_display(pooled), hide_index=True, width="stretch")
    st.download_button("形状比較のプールCSV（36行）をダウンロード", csv_bytes, file_name="rq_m04_pooled_summary.csv", mime="text/csv", key="shape_csv")
    overview = EVIDENCE / "figures" / "overview.html"
    if overview.exists():
        st.download_button("自己完結図 overview.html を保存", overview.read_bytes(), file_name="rq_m04_overview.html", mime="text/html", key="shape_overview")
    st.info("有限Bの端点安定性、区間の短さ、母集団反復の被覆は別の性質です。B4999を真値・無限Bの正解とせず、"
            "再標本化を増やしても観測情報は増えません。形状間は独立群、B差・方法差は同じ原標本内の記述的な対応差です。"
            "非有意を収束・差ゼロの証明とせず、歪度単独の因果効果・全域の単調性・無限Bの被覆へ一般化しません。")
