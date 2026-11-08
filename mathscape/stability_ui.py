"""RQ-M03 saved-result viewer. No generation or inference imports."""

import csv
import hashlib
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "rq_m03"
STATE_KEY = "stability_saved_view"
DEFAULT = dict(distribution="normal", n=5, seed=0, outer=0, B=199, k=0,
               methods=["t", "percentile", "bca"])
LABELS = {"normal": "正規分布 N(0,1)", "centered_exponential": "中心化指数分布 Exp(1)−1"}
METHODS = {"t": "t参照", "percentile": "percentile", "bca": "BCa"}
COLORS = {"t": "#46DCC0", "percentile": "#FFB86B", "bca": "#A8A2FF"}


def _state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = DEFAULT.copy()
    return st.session_state[STATE_KEY]


def _remember():
    state = _state()
    for name in DEFAULT:
        if f"stab_{name}" in st.session_state:
            state[name] = st.session_state[f"stab_{name}"]


@st.cache_data(max_entries=4, show_spinner=False)
def _panel(path, modified):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


@st.cache_data(max_entries=4, show_spinner=False)
def _table(path, modified):
    payload = Path(path).read_bytes()
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8")))), payload


def _display_rows(rows):
    output = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            try:
                number = float(value)
                converted[key] = number if np.isfinite(number) else None
            except (ValueError, TypeError):
                converted[key] = value
        output.append(converted)
    return output


def _chart(panel, state):
    oi = list(panel["outer_id"]).index(state["outer"])
    bi = list(panel["B_values"]).index(state["B"])
    sample = panel["samples"][oi]
    fig = go.Figure(go.Scatter(x=sample, y=np.full(sample.size, -4), mode="markers",
                              name="保存原標本の観測点", marker=dict(symbol="circle", size=9, color="white")))
    for mi, method in enumerate(panel["methods"]):
        if method not in state["methods"]:
            continue
        lower, upper = panel["lower"][oi, bi, mi], panel["upper"][oi, bi, mi]
        valid = panel["valid"][oi, bi, mi]
        if method == "t":
            if valid[0]:
                fig.add_scatter(x=[lower[0], upper[0]], y=[-2, -2], mode="lines+markers",
                                name="t参照（原標本から1回）", line=dict(color=COLORS[method], width=3))
            continue
        offset = -0.12 if method == "percentile" else 0.12
        x, y = [], []
        for k in np.flatnonzero(valid):
            x.extend([lower[k], upper[k], None])
            y.extend([float(k) + offset, float(k) + offset, None])
        fig.add_scatter(x=x, y=y, mode="lines", name=f"{METHODS[method]} · 全K系列",
                        line=dict(color=COLORS[method], dash="dash" if method == "percentile" else "dot"))
        selected = state["k"]
        if valid[selected]:
            fig.add_scatter(x=[lower[selected], upper[selected]], y=[selected + offset] * 2,
                            mode="lines+markers", name=f"{METHODS[method]} · 選択k={selected}",
                            line=dict(color=COLORS[method], width=5),
                            marker=dict(symbol="line-ns", size=14, line=dict(color=COLORS[method], width=2)))
    fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="真の母平均 μ=0")
    fig.update_layout(template="plotly_dark", height=570, margin=dict(l=35, r=25, t=70, b=40),
                      xaxis_title="観測値／保存95%信頼区間の端点（標準化単位）",
                      yaxis_title="内側系列ID k（原標本とt参照は下部に配置）",
                      legend=dict(orientation="h", y=1.18), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    return oi, bi


def render_stability_view():
    st.header("Inference Lab · 再標本化回数と端点の安定性")
    st.write("保存した同じ原標本に、独立なK系列で再標本化した区間を重ねる。端点の計算上の揺れと、母平均を含む頻度を分けて観察する。")
    st.info("固定α=0.05、真のμ=0・σ²=1。旧自由操作モードのαとは独立しています。このビューは保存データだけを読み、標本やbootstrap列を生成しません。")
    st.caption("n：原標本の観測数。R：セル当たり100の独立原標本。K：同じ原標本への20の独立再標本化系列。"
               "B：各系列の再標本化回数199・999・4999。同じ長さ4999の平均列のprefixを使う対応比較です。")
    state = _state()
    for name, value in state.items():
        st.session_state.setdefault(f"stab_{name}", value)
    controls = [("保存母集団", "distribution", list(LABELS)), ("保存標本数 n", "n", [5, 20]),
                ("保存master seed", "seed", [0, 1, 2]), ("保存原標本ID", "outer", list(range(100))),
                ("保存bootstrap回数 B", "B", [199, 999, 4999]), ("表示する内側系列 k", "k", list(range(20)))]
    for label, name, options in controls:
        st.sidebar.selectbox(label, options, key=f"stab_{name}", on_change=_remember,
                             format_func=LABELS.get if name == "distribution" else str)
    st.sidebar.multiselect("安定性で表示する手法", list(METHODS), format_func=METHODS.get,
                           key="stab_methods", on_change=_remember)
    st.sidebar.caption("分布・n・seed・原標本IDで元の観測データを変更します。B・内側k・手法は同じ原標本の表示だけを切り替えます。")
    filename = f"{state['distribution']}_n{state['n']}_seed{state['seed']}.npz"
    path = EVIDENCE / "panels" / filename
    try:
        panel = _panel(str(path), path.stat().st_mtime_ns)
        originals, _ = _table(str(EVIDENCE / "original_metrics.csv"), (EVIDENCE / "original_metrics.csv").stat().st_mtime_ns)
        seed_rows, _ = _table(str(EVIDENCE / "summary.csv"), (EVIDENCE / "summary.csv").stat().st_mtime_ns)
        pooled, pooled_bytes = _table(str(EVIDENCE / "pooled_summary.csv"), (EVIDENCE / "pooled_summary.csv").stat().st_mtime_ns)
        sample = panel["samples"][state["outer"]]
        digest = hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes()).hexdigest()
        if digest != panel["sample_sha256"][state["outer"]]:
            raise ValueError("保存原標本のhashが一致しません")
    except (OSError, KeyError, IndexError, ValueError) as error:
        st.error(f"保存結果を表示できません：{type(error).__name__}。RQ-M03の証拠ファイルを確認してください。")
        return
    st.success(f"表示条件：{LABELS[state['distribution']]} ／ n={state['n']} ／ master seed={state['seed']} ／ "
               f"原標本ID={state['outer']} ／ B={state['B']} ／ 選択k={state['k']} ／ 固定α=0.05")
    st.caption(f"原標本SHA-256：{digest}")
    st.subheader("1. 1つの原標本に対するK系列の端点")
    oi, bi = _chart(panel, state)
    diagnostics = []
    for mi, method in enumerate(panel["methods"]):
        if method not in state["methods"]:
            continue
        row = dict(method=str(method), B=state["B"], k=state["k"])
        for name in ("lower", "upper", "valid", "reason", "z0", "a", "q_low", "q_high", "tau_low", "tau_high", "tail_warning"):
            row[name] = panel[name][oi, bi, mi, state["k"]].item()
        diagnostics.append(row)
    st.dataframe(diagnostics, hide_index=True, width="stretch")
    invalid = [row for row in diagnostics if not row["valid"]]
    if invalid:
        st.warning("選択した区間に無効結果があります：" + " ／ ".join(f"{row['method']}: {row['reason']}" for row in invalid))
    st.caption("tau=B×min(q,1−q)。上下いずれかのtau<10を尾部解像度警告とし、除外しません。"
               "警告と揺れの関係は関連であり、警告の因果効果を意味しません。無効時の別手法への置換はありません。")
    matches = lambda row: row["distribution"] == state["distribution"] and int(row["n"]) == state["n"]
    sample_rows = [row for row in originals if matches(row) and int(row["master_seed"]) == state["seed"] and int(row["outer_id"]) == state["outer"]]
    columns = ["B", "method", "se", "lower_sd", "upper_sd", "A", "C", "warning_rate", "mean_width", "complete", "valid_inner", "invalid_inner", "total_inner"]
    st.subheader("2. この原標本のB別の端点SD・A")
    st.dataframe(_display_rows([{key: row[key] for key in columns} for row in sample_rows]), hide_index=True, width="stretch")
    st.caption("上下端点SDは同じ原標本のK系列間の標本SD（ddof=1）。A=√((SD_L²+SD_U²)/2)/se、se=s/√n。"
               "AはSE単位の有限K指標で、真値に対するRMSEや推定量の標準誤差ではありません。Cはこの原標本のK区間の被覆割合です。tは1回の参照でSDとAは0です。")
    st.subheader("3. 母集団から繰り返した全原標本の集計")
    st.write("3seedプール：独立単位は300原標本です。以下の率・幅・警告とMCSEは各原標本内のKを平均してから、原標本間で計算しています。6,000区間を独立観測とは数えません。")
    current = next((row for row in pooled if matches(row) and int(row["B"]) == state["B"] and row["method"] == "bca"), None)
    if current:
        def number(key, spec=".4g"):
            value = float(current[key])
            return format(value, spec) if np.isfinite(value) else "未定義"

        st.caption(f"選択B={state['B']}のBCa · 3seedプール300原標本（表示するk・手法の選択とは独立した全体集計）")
        columns = st.columns(3)
        columns[0].metric("BCa被覆率 · 300原標本", number("coverage", ".2%"))
        columns[1].metric("被覆率のMC標準誤差", number("coverage_mcse", ".2%"))
        columns[2].metric("平均A · 300原標本", number("A"))
    compact = ["B", "method", "original_total", "coverage", "coverage_mcse", "A", "A_mcse", "lower_sd", "upper_sd",
               "mean_width", "mean_width_mcse", "warning_rate", "warning_rate_mcse", "interval_count", "valid_count", "invalid_count", "incomplete_original_count"]
    st.dataframe(_display_rows([{key: row[key] for key in compact} for row in pooled if matches(row)]), hide_index=True, width="stretch")
    with st.expander("選択seedの100原標本の集計"):
        st.dataframe(_display_rows([row for row in seed_rows if matches(row) and int(row["master_seed"]) == state["seed"]]), hide_index=True, width="stretch")
    with st.expander("保存済みプール集計36行をすべて見る"):
        st.dataframe(_display_rows(pooled), hide_index=True, width="stretch")
    st.download_button("保存済みプールCSV（36行）をダウンロード", data=pooled_bytes,
                       file_name="rq_m03_pooled_summary.csv", mime="text/csv", key="stab_download")
    st.caption("coverageは原標本ごとのCの平均、coverage_mcseは原標本間のMCSEです。decision_low/highは原標本単位の99% t区間で、"
               "通常のWilson区間ではありません。complete/incomplete・valid/invalidの分母を保持し、無効がある主集計はNA、共同有効の参考値にはconditional_を付けます。")
    st.info("再標本化回数を増やしても観測データは増えない。端点の安定性と95%の被覆は別です。B=4999は真値や無限Bの正解ではありません。"
            "平均端点から新しい信頼区間を作って被覆を評価していません。有限Kの端点SD、区間の短さ、反復被覆を区別してください。")
