"""Read-only RQ-M05 matched-moment populations and saved interval results."""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mathscape.matched_moments import population_distribution, population_info


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "rq_m05"
STATE_KEY = "matched_saved_view"
LABELS = {"gamma": "G：標準化ガンマ", "lognormal": "L：標準化対数正規"}
METHODS = {"t": "t", "percentile": "percentile", "bca": "BCa"}
COLORS = {"gamma": "#46DCC0", "lognormal": "#FFB86B", "t": "#46DCC0", "percentile": "#FFB86B", "bca": "#A8A2FF"}
STABILITY = ["lower_sd", "upper_sd", "S", "Ase", "Apop"]


def _state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = dict(distribution="gamma", methods=list(METHODS),
                                         selections={dist: dict(seed=0, outer=0) for dist in LABELS})
    return st.session_state[STATE_KEY]


def _switch_population():
    state = _state()
    state["distribution"] = st.session_state.matched_distribution
    for key, value in state["selections"][state["distribution"]].items():
        st.session_state[f"matched_{key}"] = value


def _remember():
    state = _state()
    for key in ("seed", "outer"):
        if f"matched_{key}" in st.session_state:
            state["selections"][state["distribution"]][key] = st.session_state[f"matched_{key}"]
    if "matched_methods" in st.session_state:
        state["methods"] = st.session_state.matched_methods


@st.cache_data(max_entries=4, show_spinner=False)
def _panel(path, modified):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


@st.cache_data(max_entries=4, show_spinner=False)
def _csv(path, modified):
    payload = Path(path).read_bytes()
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8")))), payload


def _display(rows):
    converted = []
    for row in rows:
        entry = {}
        for key, value in row.items():
            if value in ("True", "False"):
                entry[key] = value == "True"
                continue
            try:
                number = float(value)
                entry[key] = number if np.isfinite(number) else None
            except (ValueError, TypeError):
                entry[key] = value
        converted.append(entry)
    return converted


def _number(value, fmt=".5g"):
    return format(float(value), fmt) if value is not None and np.isfinite(float(value)) else "未定義"


def _pp(value):
    return f"{float(value) * 100:+.3f} pp" if value is not None and np.isfinite(float(value)) else "未定義"


def _chart(figure, x, y, height=370):
    figure.update_layout(template="plotly_dark", height=height, xaxis_title=x, yaxis_title=y,
                         margin=dict(l=40, r=25, t=65, b=40), legend=dict(orientation="h", y=1.2),
                         paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _populations():
    st.subheader("1. 同じ前三モーメント、異なる分布")
    columns = st.columns(3)
    for column, label, value in zip(columns, ["共通の母平均", "共通の母分散", "共通の母歪度"], [0, 1, 2]):
        column.metric(label, value)
    populations = {dist: population_distribution(dist) for dist in LABELS}
    lower = min(float(pop.ppf(0.0005)) for pop in populations.values())
    upper = max(float(pop.ppf(0.9995)) for pop in populations.values())
    x = np.linspace(lower, upper, 1000)
    figure, facts = go.Figure(), []
    for dist, population in populations.items():
        info, y = population_info(dist), population.pdf(x)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite population display density")
        figure.add_scatter(x=x, y=y, name=LABELS[dist], mode="lines",
                           line=dict(color=COLORS[dist], width=3, dash="solid" if dist == "gamma" else "dash"))
        facts.append({"母集団": LABELS[dist], "母平均": info["mean"], "母分散": info["variance"], "母歪度": info["skewness"],
                      "超過尖度": info["excess_kurtosis"], "支持下限": info["lower_bound"],
                      "表示窓より下の確率": float(population.cdf(lower)), "表示窓より上の確率": float(population.sf(upper))})
    _chart(figure, "X（共通の標準化単位・共通表示窓）", "母集団の確率密度 f(x)", 320)
    st.dataframe(facts, hide_index=True, width="stretch")
    st.caption(f"共通表示窓は[{lower:.5g}, {upper:.5g}]。両分布の0.05%・99.95%分位点を覆う窓で、"
               "分布ごとの下側・上側省略確率を表に示しています。密度を切り取って再正規化していません。支持域外の密度は数学的に0です。")
    st.latex(r"G_{\rm raw}\sim\mathrm{Gamma}(1,1),\quad X_G=G_{\rm raw}-1,\qquad X_L=\frac{\exp(sZ)-m}{d},\quad Z\sim N(0,1)")
    st.latex(r"v=2\sinh\!\left(\frac{\operatorname{asinh}(1)}3\right),\ t=1+v^2,\ s=\sqrt{\log(1+v^2)},\ m=\sqrt t,\ d=\sqrt t\,v")
    st.info("解析的な母モーメントで標準化した2分布です。丸めた定数で生成せず、各原標本を実現平均0・分散1・歪度2へ強制調整していません。"
            "推論対象はXの算術平均で、対数平均ではありません。尖度・支持域・尾などは異なるため、差を尖度だけの因果効果とは呼べません。")


def _sample_csv(sample, dist, seed, outer, digest):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["distribution", "master_seed", "outer_id", "observation_index", "X", "sample_sha256"])
    for index, value in enumerate(sample):
        writer.writerow([dist, seed, outer, index, float(value), digest])
    return buffer.getvalue().encode("utf-8")


def _sample(panel, state, selection):
    outer, sample = selection["outer"], panel["samples"][selection["outer"]]
    figure = go.Figure(go.Scatter(x=sample, y=np.zeros(20), mode="markers", name="保存原標本20観測",
                                 marker=dict(symbol="circle", size=8, color="white", line=dict(color="#080C14", width=1))))
    rows = []
    for mi, method in enumerate(panel["methods"]):
        if method not in state["methods"]:
            continue
        valid = bool(panel["valid"][outer, mi])
        code = int(panel["reason_codes"][outer, mi])
        reason = str(panel["reason_labels"][code])
        if valid:
            figure.add_scatter(x=[panel["lower"][outer, mi], panel["upper"][outer, mi]], y=[mi + 1] * 2,
                               mode="lines+markers", name=METHODS[method], line=dict(color=COLORS[method], width=4, dash=["solid", "dash", "dot"][mi]),
                               marker=dict(symbol="line-ns", size=15, line=dict(color=COLORS[method], width=2)))
        else:
            st.warning(f"{METHODS[method]}は無効：{reason}。別手法への置換は行いません。")
        row = dict(method=str(method), valid=valid, reason=reason)
        row.update({key: panel[key][outer, mi].item() for key in ("lower", "upper", "z0", "a", "q_low", "q_high", "tau_low", "tau_high", "tail_warning")})
        row.update({key: None for key in STABILITY})
        rows.append(row)
    figure.add_vline(x=0, line_dash="dot", line_color="white", annotation_text="真の算術平均 μ=0")
    figure.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["原標本", "t", "percentile", "BCa"], range=[-0.5, 3.6])
    _chart(figure, "原標本観測値／母平均の95%推論区間（標準化単位）", "同じ原標本に適用した3方法", 380)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("percentileとBCaは同じ4,999個のbootstrap平均を使っています。端点tau<10の警告は除外規則ではありません。"
               "95%は反復手続きの名目水準で、実現した一つの区間に真値が95%の確率で存在するという意味ではありません。")


def _summaries(pooled):
    st.subheader("3. 全2分布×3方法：独立原標本からの被覆と方向")
    st.write("各分布N=30,000（3seed×R10,000）。独立な単位は原標本とその1回のbootstrap計算の組です。B=4,999を母集団標本数へ数えません。"
             "以下は全反復の保存集計で、上の原標本1例から推定した性能ではありません。")
    figure = go.Figure()
    for dist in LABELS:
        rows = [next(row for row in pooled if row["distribution"] == dist and row["method"] == method) for method in METHODS]
        values = np.array([float(row["C"]) for row in rows])
        figure.add_scatter(x=list(METHODS.values()), y=values, mode="markers+lines", name=LABELS[dist], line_color=COLORS[dist],
                           marker_symbol="circle" if dist == "gamma" else "diamond",
                           error_y=dict(type="data", symmetric=False, array=np.array([float(row["C_decision_high"]) for row in rows]) - values,
                                        arrayminus=values - np.array([float(row["C_decision_low"]) for row in rows])))
    figure.add_hline(y=0.95, line_dash="dot", annotation_text="名目95%")
    _chart(figure, "保存済み推論方法", "被覆率Cと99% Monte Carlo区間", 360)
    columns = ["distribution", "method", "original_total", "C", "C_mcse", "C_decision_low", "C_decision_high", "Mbelow", "Mabove", "T", "T_mcse",
               "mean_width", "mean_width_mcse", "warning_rate", "valid_count", "invalid_count", "missing_original_count"]
    st.dataframe(_display([{key: row[key] for key in columns} for row in pooled]), hide_index=True, width="stretch")
    if any(int(row["invalid_count"]) or int(row["missing_original_count"]) for row in pooled):
        st.warning("無効・欠測を含む主集計は未定義です。conditional_の参考値へ置換せず、分母と科学レポートを確認してください。")
    st.caption("Mbelow=I(U<0)：区間全体が真値より下。Mabove=I(L>0)：区間全体が真値より上。T=Mbelow−Mabove、C+Mbelow+Mabove=1です。"
               "99% Monte Carlo区間は性能推定の不確実性で、原標本から作る95%推論区間とは別です。")


def _hypotheses(hypotheses, pooled):
    st.subheader("4. BCa主比較：差と±1ppの事前許容帯")
    quantities = hypotheses["quantities"]
    figure = go.Figure()
    for name, label in (("H1", "被覆差 ΔC"), ("H2", "方向差 ΔT")):
        row = quantities[name]
        if row["difference"] is not None and row["low"] is not None and row["high"] is not None:
            figure.add_scatter(x=[row["difference"]], y=[label], mode="markers", name=name,
                               error_x=dict(type="data", symmetric=False, array=[row["high"] - row["difference"]], arrayminus=[row["difference"] - row["low"]]))
    figure.add_vrect(x0=-0.01, x1=0.01, fillcolor="#46DCC0", opacity=0.13, line_width=0, annotation_text="±1pp帯")
    figure.add_vline(x=0, line_dash="dot", line_color="white")
    figure.update_xaxes(tickformat=".1%")
    _chart(figure, "L−Gの独立2群差と99% Welch区間（目盛1%=1pp、各群N=30,000）", "BCa・n20・B4999・K1・α.05固定", 300)
    columns = st.columns(2)
    for column, name, label in zip(columns, ["H1", "H2"], ["BCa被覆差 ΔC（L−G）", "BCa方向差 ΔT（L−G）"]):
        column.metric(label, _pp(quantities[name]["difference"]))
    for name in ("H1", "H2"):
        row = quantities[name]
        scope = "被覆C" if name == "H1" else "方向差T"
        if row["equivalence_supported"]:
            status = f"実用差仮説は不支持。この2分布・n20・α.05・B4999・BCaの{scope}について±1pp内の実用的同等性を支持"
        elif row["status"] == "supported":
            status = "1pp超の実用差を支持"
        else:
            status = "保留（閾値への接触・不確実性・必要データの無効等）"
        st.write(f"{name}：{status}。99%差区間 [{_pp(row['low'])}, {_pp(row['high'])}]。")
    if hypotheses["both_metrics_equivalent"]:
        st.success("両指標で±1pp内を支持。ただし、推論区間の被覆が名目95%を満たすという意味ではありません。")
    coverage = {row["distribution"]: row for row in pooled if row["method"] == "bca"}
    columns = st.columns(2)
    columns[0].metric("BCa被覆 G", _number(coverage["gamma"]["C"], ".3%"))
    columns[1].metric("BCa被覆 L", _number(coverage["lognormal"]["C"], ".3%"))
    st.caption("1pp=.01の絶対的な確率差で、相対的な増減率の%ではありません。差の図の目盛1%=1ppです。"
               "差0の棄却と1pp超の差は別です。99%区間が厳密に±1pp内に含まれることを同等性支持の条件とし、接触は保留です。"
               "非有意だけでは同等とは言いません。seedは管理ブロックで、全seed同符号という追加ゲートはありません。")
    seed_rows = []
    for row in hypotheses["seed_details"]:
        for name in ("H1", "H2"):
            seed_rows.append(dict(master_seed=row["master_seed"], 指標=name, 差=row[name]["difference"], 下限99=row[name]["low"], 上限99=row[name]["high"]))
    with st.expander("3seedの差と99%区間（記述用、方向ゲートではありません）"):
        st.dataframe(seed_rows, hide_index=True, width="stretch")


def render_matched_view():
    st.header("Inference Lab · 同じ歪度、違う分布")
    st.caption("保存結果専用 · n=20、α=0.05、B=4,999、K=1固定。旧自由操作のαから独立しています。生成・bootstrap・主実験は起動しません。")
    state = _state()
    selection = state["selections"][state["distribution"]]
    for key, value in dict(distribution=state["distribution"], methods=state["methods"], **selection).items():
        st.session_state.setdefault(f"matched_{key}", value)
    st.sidebar.selectbox("一致モーメントの母集団", list(LABELS), format_func=LABELS.get, key="matched_distribution", on_change=_switch_population)
    st.sidebar.selectbox("一致モーメントのmaster seed", [0, 1, 2], key="matched_seed", on_change=_remember)
    st.sidebar.number_input("一致モーメントの原標本ID", 0, 9999, step=1, key="matched_outer", on_change=_remember)
    st.sidebar.multiselect("一致モーメントで表示する手法", list(METHODS), format_func=METHODS.get, key="matched_methods", on_change=_remember)
    st.sidebar.caption("各分布の初期例はseed0/outer0。分布切替は独立した別原標本への切替で、同じIDでも対応標本ではありません。戻ると分布別の選択IDを復元します。")
    selection = state["selections"][state["distribution"]]
    try:
        path = EVIDENCE / "panels" / f"{state['distribution']}_seed{selection['seed']}.npz"
        panel = _panel(str(path), path.stat().st_mtime_ns)
        pooled, csv_bytes = _csv(str(EVIDENCE / "pooled_summary.csv"), (EVIDENCE / "pooled_summary.csv").stat().st_mtime_ns)
        seeds, _ = _csv(str(EVIDENCE / "summary.csv"), (EVIDENCE / "summary.csv").stat().st_mtime_ns)
        hypotheses = json.loads((EVIDENCE / "hypotheses.json").read_text())
        sample = panel["samples"][selection["outer"]]
        digest = hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes()).hexdigest()
        expected_hash = panel["sample_sha256"][selection["outer"]].item()
        expected_hash = expected_hash.decode("ascii") if isinstance(expected_hash, bytes) else str(expected_hash)
        if digest != expected_hash:
            raise ValueError("saved sample hash mismatch")
        _populations()
    except (OSError, ValueError, KeyError, IndexError) as error:
        st.error(f"RQ-M05の保存結果を表示できません：{type(error).__name__}。証拠ファイルを確認してください。")
        return
    st.success(f"表示条件：{LABELS[state['distribution']]} ／ seed={selection['seed']} ／ 原標本ID={selection['outer']} ／ n20・固定α=.05・B4999・K1")
    st.caption(f"原標本SHA-256：{digest}")
    st.subheader("2. 保存原標本と同じ標本からの3区間")
    _sample(panel, state, selection)
    st.info("K=1のため端点SD_L・SD_U・S・Ase・Apopは全方法（tも含む）でNA／対象外です。0で埋めず、"
            "端点の条件付き計算変動を推定しません。旧RQ-M03/M04の安定性表示とは実験配分が異なります。")
    _summaries(pooled)
    _hypotheses(hypotheses, pooled)
    with st.expander("保存済みプール6行（conditional_参考値・NA・全分母）"):
        st.dataframe(_display(pooled), hide_index=True, width="stretch")
    with st.expander("全3seed・2分布・3方法の18行"):
        st.dataframe(_display(seeds), hide_index=True, width="stretch")
    st.download_button("プール要約CSV（6行）をダウンロード", csv_bytes, file_name="rq_m05_pooled_summary.csv", mime="text/csv", key="matched_summary_csv")
    st.download_button("選択原標本CSV（20観測）をダウンロード", _sample_csv(sample, state["distribution"], selection["seed"], selection["outer"], digest),
                       file_name=f"rq_m05_{state['distribution']}_seed{selection['seed']}_outer{selection['outer']}.csv", mime="text/csv", key="matched_sample_csv")
    overview = EVIDENCE / "figures" / "overview.html"
    if overview.exists():
        st.download_button("自己完結図 overview.html を保存", overview.read_bytes(), file_name="rq_m05_overview.html", mime="text/html", key="matched_overview")
    st.info("結論は事前選定したこの2分布・固定条件・有限B手続き・該当指標に限定されます。前三モーメントが全分布で十分とは証明せず、"
            "尖度だけの効果、無限Bの性能、名目95%への校正保証へ拡張しません。conditional_の有効例参考値は主判定を置き換えません。")
