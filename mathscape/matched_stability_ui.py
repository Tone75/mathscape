"""Read-only RQ-M06 view: reused originals and saved K30 endpoint variation."""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "rq_m06"
M05 = ROOT / "evidence" / "rq_m05"
STATE_KEY = "matched_stability_saved_view"
LABELS = {"gamma": "G：標準化ガンマ", "lognormal": "L：標準化対数正規"}
METHODS = {"t": "t", "percentile": "percentile", "bca": "BCa"}
COLORS = {"t": "#46DCC0", "percentile": "#FFB86B", "bca": "#A8A2FF"}


def _state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = dict(distribution="gamma", B=4999, methods=list(METHODS),
                                         selections={dist: dict(seed=0, outer=0, k=0) for dist in LABELS})
    return st.session_state[STATE_KEY]


def _switch_population():
    state = _state()
    state["distribution"] = st.session_state.mst_distribution
    for name, value in state["selections"][state["distribution"]].items():
        st.session_state[f"mst_{name}"] = value


def _remember():
    state = _state()
    for name in ("seed", "outer", "k"):
        if f"mst_{name}" in st.session_state:
            state["selections"][state["distribution"]][name] = st.session_state[f"mst_{name}"]
    for name in ("B", "methods"):
        if f"mst_{name}" in st.session_state:
            state[name] = st.session_state[f"mst_{name}"]


@st.cache_data(max_entries=4, show_spinner=False)
def _panel(path, modified):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


@st.cache_data(max_entries=8, show_spinner=False)
def _csv(path, modified):
    payload = Path(path).read_bytes()
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8")))), payload


@st.cache_data(max_entries=4, show_spinner=False)
def _old_hashes(path, modified):
    with np.load(path, allow_pickle=False) as saved:
        return saved["sample_sha256"].astype(str)


def _display(rows):
    converted = []
    for row in rows:
        result = {}
        for key, value in row.items():
            if value in ("True", "False"):
                result[key] = value == "True"
                continue
            try:
                number = float(value)
                result[key] = number if np.isfinite(number) else None
            except (TypeError, ValueError):
                result[key] = value
        converted.append(result)
    return converted


def _number(value, fmt=".6g"):
    return format(float(value), fmt) if value is not None and np.isfinite(float(value)) else "未定義"


def _chart(figure, x, y, height=360):
    figure.update_layout(template="plotly_dark", height=height, xaxis_title=x, yaxis_title=y,
                         margin=dict(l=40, r=25, t=70, b=40), legend=dict(orientation="h", y=1.2),
                         paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _intervals(panel, state, selection):
    outer, k = selection["outer"], selection["k"]
    bi = list(panel["B_values"]).index(state["B"])
    sample = panel["samples"][outer]
    figure = go.Figure(go.Scatter(x=sample, y=np.full(sample.size, -4), mode="markers", name="固定した保存原標本",
                                 marker=dict(symbol="circle", size=8, color="white", line=dict(color="#080C14", width=1))))
    records = []
    for mi, method in enumerate(panel["methods"]):
        if method not in state["methods"]:
            continue
        lower, upper, valid = (panel[name][outer, bi, mi] for name in ("lower", "upper", "valid"))
        if method == "t":
            if valid[0]:
                figure.add_scatter(x=[lower[0], upper[0]], y=[-2, -2], name="t参照（1回）", mode="lines+markers", line_color=COLORS[method])
        else:
            offset = -0.13 if method == "percentile" else 0.13
            x, y = [], []
            for inner in np.flatnonzero(valid):
                x.extend([lower[inner], upper[inner], None])
                y.extend([float(inner) + offset] * 2 + [None])
            figure.add_scatter(x=x, y=y, name=f"{METHODS[method]} · 全K30", mode="lines",
                               line=dict(color=COLORS[method], dash="dash" if method == "percentile" else "dot"))
            if valid[k]:
                figure.add_scatter(x=[lower[k], upper[k]], y=[k + offset] * 2, mode="lines+markers", name=f"{METHODS[method]} · 選択k={k}",
                                   line=dict(color=COLORS[method], width=5), marker=dict(symbol="line-ns", size=14, line=dict(color=COLORS[method], width=2)))
        reason = str(panel["reason_labels"][int(panel["reason_codes"][outer, bi, mi, k])])
        row = dict(method=str(method), B=state["B"], inner_id=k, reason=reason)
        row.update({name: panel[name][outer, bi, mi, k].item() for name in ("lower", "upper", "valid", "z0", "a", "q_low", "q_high", "tau_low", "tau_high", "tail_warning")})
        records.append(row)
        if not row["valid"]:
            st.warning(f"{METHODS[method]}は無効：{reason}。別手法への置換はありません。")
    # Preserve a common physical axis when B or visible methods change.
    all_x = np.concatenate([sample, panel["lower"][outer].ravel(), panel["upper"][outer].ravel()])
    all_x = all_x[np.isfinite(all_x)]
    low, high = float(all_x.min()), float(all_x.max())
    padding = max((high - low) * .05, 1e-6)
    figure.update_xaxes(range=[low - padding, high + padding], autorange=False)
    figure.add_vline(x=0, line_dash="dot", line_color="white", annotation_text="真の平均 μ=0")
    _chart(figure, "保存95%推論区間の端点／観測値（標準化単位、全B共通軸）", "内側系列ID k（下部：原標本とt参照）", 560)
    st.dataframe(records, hide_index=True, width="stretch")
    st.caption("同じ原標本の全B・全K・全手法・観測値から横軸範囲を固定しています。Bや手法表示だけで軸を拡大縮小しません。"
               "選択kを太く表示し、tau=B×min(q,1−q)<10の警告は除外しません。95%推論区間はこの標本からの手続きで、実現区間に真値が95%の確率であるという意味ではありません。")


def _original_metrics(rows, b):
    current = next(row for row in rows if row["method"] == "bca" and int(row["B"]) == b)
    st.caption(f"選択原標本・BCa・B={b}のK30系列から算出した保存指標。選択kや表示する手法の選択とは独立しています。")
    columns = st.columns(3)
    for column, label, key in zip(columns, ["下端点SD", "上端点SD", "合成端点変動 S"], ["lower_sd", "upper_sd", "S"]):
        column.metric(label, _number(current[key]))
    columns = st.columns(3)
    for column, label, key in zip(columns, ["標本SE（s/√20）", "原標本Ase", "原標本Apop"], ["se", "Ase", "Apop"]):
        column.metric(label, _number(current[key]))
    fields = ["B", "method", "se", "lower_sd", "upper_sd", "S", "Ase", "Apop", "Vse", "mean_width", "warning_rate", "C", "T", "complete", "valid_inner", "invalid_inner", "total_inner"]
    st.dataframe(_display([{field: row[field] for field in fields} for row in rows]), hide_index=True, width="stretch")
    st.info("下端点SD・上端点SDは原標本内K30でddof=1。S=hypot(SD_L,SD_U)/√2、Ase=S/(s/√20)、Apop=S/(1/√20)、Vse=Ase²です。"
            "端点の再計算による揺れであり、bootstrap平均のSD・平均推定のSE・真値へのRMSEではありません。t対照はゼロ変動です。"
            "有限K30でsqrtを含む測定量なので、理想的な条件付きSDの不偏推定とは主張しません。")


def _pooled(pooled):
    st.subheader("3. 独立原標本N=600／分布の全条件集計")
    figure = go.Figure()
    for dist, color in (("gamma", "#46DCC0"), ("lognormal", "#FFB86B")):
        rows = [next(row for row in pooled if row["distribution"] == dist and row["method"] == "bca" and int(row["B"]) == b) for b in (999, 4999)]
        means = np.array([float(row["Ase"]) for row in rows])
        figure.add_scatter(x=["999", "4999"], y=means, mode="lines+markers", name=LABELS[dist], line_color=color,
                           error_y=dict(type="data", symmetric=False, array=np.array([float(row["Ase_decision_high"]) for row in rows]) - means,
                                        arrayminus=means - np.array([float(row["Ase_decision_low"]) for row in rows])))
    _chart(figure, "共通prefixのbootstrap回数 B（BCa）", "平均Aseと原標本単位99%比較区間", 330)
    compact = ["distribution", "B", "method", "original_total", "Ase", "Ase_mcse", "Ase_p05", "Ase_p50", "Ase_p95", "Apop", "S", "lower_sd", "upper_sd", "Vse", "mean_width", "warning_rate", "C", "T", "interval_count", "valid_count", "invalid_count", "missing_original_count"]
    st.dataframe(_display([{key: row[key] for key in compact} for row in pooled]), hide_index=True, width="stretch")
    st.caption("全12行。各原標本内の量を計算してから原標本間で集計し、K30やBを独立原標本数に数えません。"
               "p05/p50/p95等は保存済み原標本指標のlinear分位点です。invalid/missingがあれば主値はNA、conditional_は参考値です。")
    st.info("99%区間は、結果によらずID選択した保存IID原標本に関するモデルベースの原標本間近似区間です。"
            "有限K30の測定ノイズを含み、この固定1,200標本を条件とする内側乱数だけの区間ではありません。"
            "M05保存パネルの再利用であり、新しい独立追試ではありません。C/Tは部分パネルの記述値だけで、M05の同等性判断を再検定・更新・プールしません。")


def _hypotheses(hypotheses):
    st.subheader("4. 事前固定した安定性の比較")
    q = hypotheses["quantities"]
    h1 = q["H1"]
    figure = go.Figure()
    if h1["difference"] is not None and h1["low"] is not None and h1["high"] is not None:
        figure.add_scatter(x=[h1["difference"]], y=["Δ平均Ase：L−G"], mode="markers", name="H1：99% Welch区間",
                           error_x=dict(type="data", symmetric=False, array=[h1["high"] - h1["difference"]], arrayminus=[h1["difference"] - h1["low"]]))
    figure.add_vrect(x0=-.01, x1=.01, opacity=.13, fillcolor="#46DCC0", line_width=0, annotation_text="±.01 Ase単位")
    figure.add_vline(x=0, line_dash="dot", line_color="white")
    _chart(figure, "平均Aseの独立2群差（各群N600・BCa・B4999）", "H1：端点変動の実用差", 260)
    st.metric("H1 平均Ase差 L−G", _number(h1["difference"], "+.7f"))
    status = "実用差仮説は不支持、固定条件の平均Ase差について±.01内の実用的同等性を支持" if h1["equivalence_supported"] else ".01超の実用差を支持" if h1["status"] == "supported" else "保留"
    st.write(f"H1：{status}。99%区間 [{_number(h1['low'], '+.7f')}, {_number(h1['high'], '+.7f')}]。")
    if h1["equivalence_supported"]:
        st.caption("今回の区間は0を除外していても、事前の±.01帯に含まれます。差0の検定と、実用差・同等性の判断は別です。")
    st.info(".01は各原標本の標本SEで規格化した端点変動量の差で、被覆率の1ppではありません。"
            "この平均差の同等性は個別原標本の安定性保証でも、名目95%の被覆保証でもありません。")
    figure = go.Figure()
    rows = []
    columns = st.columns(2)
    for column, dist in zip(columns, LABELS):
        row = q[f"H2_{dist}"]
        column.metric(f"{dist} 平均A4999/平均A999", _number(row["ratio_mean_Ase_4999_over_999"], ".6f"))
        if row["mean"] is not None and row["low"] is not None and row["high"] is not None:
            figure.add_scatter(x=[row["mean"]], y=[LABELS[dist]], name=LABELS[dist], mode="markers",
                               error_x=dict(type="data", symmetric=False, array=[row["high"] - row["mean"]], arrayminus=[row["mean"] - row["low"]]))
        rows.append(dict(母集団=LABELS[dist], N=row["N"], mean_D=row["mean"], lower99=row["low"], upper99=row["high"],
                         ratio_of_means=row["ratio_mean_Ase_4999_over_999"], 判定=row["status"]))
    figure.add_vline(x=0, line_dash="dot", line_color="white")
    _chart(figure, "D=Ase4999−.75×Ase999 の原標本単位平均と99% t区間", "H2：分布ごとの対応比較", 270)
    st.dataframe(rows, hide_index=True, width="stretch")
    overall = hypotheses["hypotheses"]["H2"]["status"]
    if overall == "supported":
        st.success("H2：両分布で平均の端点変動が25%超低減する仮説を支持。両99%区間の上限が0未満です。")
    else:
        st.warning("H2全体：" + ("不支持（少なくとも一方の分布が不支持）" if overall == "not_supported" else "保留。必要データ・各分布の区間を確認してください。"))
    st.caption("比は平均A4999/平均A999で、各原標本の比を平均した値ではありません。平均の低減は全原標本の低減を意味せず、"
               "被覆改善・真の端点への接近・偏り消失・無限B収束を意味しません。H2は上限<0で支持、下限≥0で不支持、その他は保留です。"
               "計3本の近似99%区間、Bonferroni和の非被覆率約3%以下という説明で、厳密な有限標本保証とはしません。追加のseed符号ゲートはありません。")


def _sample_csv(sample, state, selection, digest):
    buffer = io.StringIO(); writer = csv.writer(buffer)
    writer.writerow(["distribution", "master_seed", "outer_id", "observation_index", "X", "sample_sha256", "source_question"])
    for index, value in enumerate(sample):
        writer.writerow([state["distribution"], selection["seed"], selection["outer"], index, float(value), digest, "RQ-M05"])
    return buffer.getvalue().encode()


def _endpoint_csv(panel, state, selection, digest):
    buffer = io.StringIO()
    fields = ["distribution", "master_seed", "outer_id", "B", "method", "inner_id", "sample_sha256", "lower", "upper", "valid", "reason", "z0", "a", "q_low", "q_high", "tau_low", "tau_high", "tail_warning"]
    writer = csv.DictWriter(buffer, fieldnames=fields); writer.writeheader()
    for bi, b in enumerate(panel["B_values"]):
        for mi, method in enumerate(panel["methods"]):
            for k in panel["inner_id"]:
                oi = selection["outer"]
                row = dict(distribution=state["distribution"], master_seed=selection["seed"], outer_id=oi, B=int(b), method=str(method),
                           inner_id=int(k), sample_sha256=digest, reason=str(panel["reason_labels"][int(panel["reason_codes"][oi, bi, mi, k])]))
                row.update({field: panel[field][oi, bi, mi, k].item() for field in fields[7:] if field != "reason"})
                writer.writerow(row)
    return buffer.getvalue().encode()


def render_matched_stability_view():
    st.header("Inference Lab · 同じ標本でも端点は揺れる")
    st.caption("保存結果専用 · n20・固定α=.05・K30、B999/4999。M05保存原標本の各分布×seed0,1,2×outer0..199を固定再利用し、原標本を追加生成していません。")
    state = _state(); selection = state["selections"][state["distribution"]]
    for key, value in dict(distribution=state["distribution"], B=state["B"], methods=state["methods"], **selection).items():
        st.session_state.setdefault(f"mst_{key}", value)
    st.sidebar.selectbox("端点変動の母集団", list(LABELS), format_func=LABELS.get, key="mst_distribution", on_change=_switch_population)
    st.sidebar.selectbox("端点変動のmaster seed", [0, 1, 2], key="mst_seed", on_change=_remember)
    st.sidebar.number_input("端点変動の原標本ID", 0, 199, step=1, key="mst_outer", on_change=_remember)
    st.sidebar.selectbox("端点変動のbootstrap回数 B", [999, 4999], key="mst_B", on_change=_remember)
    st.sidebar.selectbox("端点変動の内側系列 k", list(range(30)), key="mst_k", on_change=_remember)
    st.sidebar.multiselect("端点変動で表示する手法", list(METHODS), format_func=METHODS.get, key="mst_methods", on_change=_remember)
    st.sidebar.caption("分布ごとのseed・原標本ID・kを復元します。分布変更は独立した別原標本への切替で、同じIDでも分布間の対応標本ではありません。表示操作で再標本化しません。")
    selection = state["selections"][state["distribution"]]
    try:
        path = EVIDENCE / "panels" / f"{state['distribution']}_seed{selection['seed']}.npz"
        panel = _panel(str(path), path.stat().st_mtime_ns)
        pooled, pooled_bytes = _csv(str(EVIDENCE / "pooled_summary.csv"), (EVIDENCE / "pooled_summary.csv").stat().st_mtime_ns)
        originals, _ = _csv(str(EVIDENCE / "original_metrics.csv"), (EVIDENCE / "original_metrics.csv").stat().st_mtime_ns)
        hypotheses = json.loads((EVIDENCE / "hypotheses.json").read_text())
        old_path = M05 / "panels" / path.name
        old_hash = _old_hashes(str(old_path), old_path.stat().st_mtime_ns)[selection["outer"]]
        sample = panel["samples"][selection["outer"]]
        digest = hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes()).hexdigest()
        saved_hash = panel["sample_sha256"][selection["outer"]].decode("ascii")
        if digest != saved_hash or digest != old_hash:
            raise ValueError("saved sample/M05 origin hash mismatch")
        old_rows, _ = _csv(str(M05 / "pooled_summary.csv"), (M05 / "pooled_summary.csv").stat().st_mtime_ns)
    except (OSError, KeyError, ValueError, IndexError) as error:
        st.error(f"RQ-M06保存結果／M05入力対応を表示できません：{type(error).__name__}。証拠ファイルを確認してください。")
        return
    st.success(f"表示条件：{LABELS[state['distribution']]} ／ seed={selection['seed']} ／ 原標本ID={selection['outer']} ／ B={state['B']} ／ k={selection['k']} ／ n20・固定α=.05・K30")
    st.caption(f"原標本SHA-256：{digest}。M05保存hashと一致。出典：evidence/rq_m05/panels/{path.name}、同じouter ID。受入入力HEAD：{panel['input_accepted_head'].item()}")
    st.subheader("1. 固定した同じ原標本の端点群と個別区間")
    _intervals(panel, state, selection)
    st.subheader("2. この原標本の物理的な揺れと規格化量")
    selected_rows = [row for row in originals if row["distribution"] == state["distribution"] and int(row["master_seed"]) == selection["seed"] and int(row["outer_id"]) == selection["outer"]]
    _original_metrics(selected_rows, state["B"])
    _pooled(pooled); _hypotheses(hypotheses)
    with st.expander("保存済みプール全12行（全分位点・conditional_・分母）"):
        st.dataframe(_display(pooled), hide_index=True, width="stretch")
    with st.expander("M05のK1安定性NAを確認（旧結果は更新しません）"):
        fields = ["distribution", "method", "K", "C", "T", "lower_sd", "upper_sd", "S", "Ase", "Apop"]
        st.dataframe(_display([{key: row[key] for key in fields} for row in old_rows if row["distribution"] == state["distribution"]]), hide_index=True, width="stretch")
        st.caption("M05 K1ではtを含め端点変動はNA。今回M06 K30のtゼロ変動と区別しています。M05のC/T判断も維持します。")
    st.download_button("端点変動プールCSV（12行）をダウンロード", pooled_bytes, file_name="rq_m06_pooled_summary.csv", mime="text/csv", key="mst_summary_csv")
    prefix = f"rq_m06_{state['distribution']}_seed{selection['seed']}_outer{selection['outer']}"
    st.download_button("固定原標本CSV（20観測）をダウンロード", _sample_csv(sample, state, selection, digest), file_name=prefix + "_sample.csv", mime="text/csv", key="mst_sample_csv")
    st.download_button("同じ原標本の全端点CSV（180行）をダウンロード", _endpoint_csv(panel, state, selection, digest), file_name=prefix + "_endpoints.csv", mime="text/csv", key="mst_endpoints_csv")
    st.caption("端点CSVは両B×3方法×K30の180行です。tのK30行は同じ参照のbroadcastで、実質1区間／Bとして集計します。"
               "自己完結HTMLはBCaの両分布・両Bの4条件とH1の概要です。H2と全12行は画面・要約CSVで確認できます。")
    overview = EVIDENCE / "figures" / "overview.html"
    if overview.exists():
        st.download_button("自己完結図 overview.html を保存", overview.read_bytes(), file_name="rq_m06_overview.html", mime="text/html", key="mst_overview")
