"""Read-only fixed evidence and explicit single-sample bootstrap exploration."""

import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mathscape.bootstrap import draw_replicate, intervals


DEFAULT = dict(dist="centered_exponential", n=5, alpha=0.05, master_seed=0, replicate_id=0)
STATE_KEY = "bootstrap_comparison"
METHODS = {"t": "t", "percentile": "percentile bootstrap", "bca": "BCa"}
COLORS = {"t": "#46DCC0", "percentile": "#FFB86B", "bca": "#A8A2FF"}
DASHES = {"t": "solid", "percentile": "dash", "bca": "dot"}
DISTRIBUTIONS = {"normal": "正規分布 N(0,1)", "centered_exponential": "中心化指数分布 Exp(1)−1"}
EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "evidence" / "rq_m02"


def _state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {
            "pending": DEFAULT.copy(), "applied": None,
            "methods": list(METHODS), "fixed_seed": 0,
        }
    return st.session_state[STATE_KEY]


def _remember_pending():
    for name in DEFAULT:
        key = f"boot_{name}"
        if key in st.session_state:
            _state()["pending"][name] = st.session_state[key]


def _remember_methods():
    _state()["methods"] = list(st.session_state.boot_methods)


def _remember_fixed_seed():
    _state()["fixed_seed"] = st.session_state.boot_fixed_seed


def _execute(new_trial=False):
    state = _state()
    config = (state["applied"] if new_trial else state["pending"]).copy()
    if new_trial:
        config["replicate_id"] += 1
    generation = {key: value for key, value in config.items() if key != "alpha"}
    if generation != state.get("generation"):
        sample, means = draw_replicate(
            generation["dist"], generation["n"], master_seed=generation["master_seed"],
            replicate_id=generation["replicate_id"], B_boot=999,
        )
    else:
        sample, means = state["sample"], state["bootstrap_means"]
    results = intervals(sample, means, alpha=config["alpha"])
    state.update(applied=config, generation=generation, sample=sample,
                 bootstrap_means=means, intervals=results)


def _new_trial():
    state = _state()
    if state["applied"] is None:
        return
    pending_matches = state["pending"] == state["applied"]
    _execute(new_trial=True)
    if pending_matches:
        state["pending"] = state["applied"].copy()
        st.session_state.boot_replicate_id = state["applied"]["replicate_id"]


def _chart(fig, x_title, y_title, height=390):
    fig.update_layout(template="plotly_dark", height=height,
                      margin=dict(l=40, r=25, t=35, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title=x_title, yaxis_title=y_title,
                      legend=dict(orientation="h", y=1.16))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def _conditions(config):
    return (f"{DISTRIBUTIONS[config['dist']]} ／ n={config['n']} ／ α={config['alpha']:g} ／ "
            f"master seed={config['master_seed']} ／ 試行ID={config['replicate_id']} ／ B_boot=999")


def _sample_view(state):
    config, sample, means, results = (state[key] for key in ("applied", "sample", "bootstrap_means", "intervals"))
    st.subheader("1. 同じ原標本から作る3つの母平均の区間")
    fig = go.Figure(go.Scatter(x=sample, y=np.zeros(sample.size), mode="markers",
                              name="原標本の観測点", marker=dict(symbol="line-ns", size=18, color="white")))
    for row, method in enumerate(METHODS, 1):
        if method not in state["methods"]:
            continue
        result = results[method]
        if result["valid"]:
            fig.add_scatter(x=[result["lower"], result["upper"]], y=[row, row], mode="lines+markers",
                            name=METHODS[method], line=dict(color=COLORS[method], dash=DASHES[method], width=4),
                            marker=dict(symbol="line-ns", size=13))
        else:
            st.warning(f"{METHODS[method]}区間は無効：{result['reason']}。別手法への置換は行いません。")
        if result["tail_warning"]:
            st.warning(f"{METHODS[method]}：有限のB_bootでは調整尾部の解像度が低い条件です。"
                       f"尾部の実効反復数は下側{result['tail_count_low']:.3g}、上側{result['tail_count_high']:.3g}"
                       "（10未満を警告）。分位点のクリップや反復追加は行いません。")
    fig.add_vline(x=0, line_dash="dash", line_color="#46DCC0", annotation_text="真の母平均 μ=0", annotation_position="top left")
    fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["原標本", *METHODS.values()], range=[-0.5, 3.6])
    _chart(fig, "観測値／母平均の信頼区間（標準化単位）", "同一原標本に適用した方法")
    cols = st.columns(3)
    cols[0].metric("標本平均", f"{np.mean(sample):.5g}")
    cols[1].metric("標本標準偏差 s（ddof=1）", f"{np.std(sample, ddof=1):.5g}")
    cols[2].metric("原標本の観測数 n", sample.size)
    st.caption(f"この標本への信頼水準は{100 * (1 - config['alpha']):g}%です。推定対象は母平均μ。"
               "真値0が分かるのはシミュレーターだからです。短い区間だけで方法の優劣は判断できません。")
    selected = [method for method in METHODS if method in state["methods"]]
    st.dataframe([{"方法": METHODS[method], "下限": results[method]["lower"], "上限": results[method]["upper"],
                   "有効": results[method]["valid"], "無効理由": results[method]["reason"],
                   "z₀": results[method]["z0"], "加速係数a": results[method]["a"],
                   "下側分位点q": results[method]["q_low"], "上側分位点q": results[method]["q_high"],
                   "尾部解像度警告": results[method]["tail_warning"]}
                  for method in selected], hide_index=True, width="stretch")
    st.subheader("2. 同じ原標本を再標本化した平均の分布")
    hist = go.Figure(go.Histogram(x=means, nbinsx=40, name="bootstrap平均（999個）", marker_color="#A8A2FF"))
    hist.add_vline(x=float(np.mean(sample)), line_dash="dot", line_color="white", annotation_text="元の標本平均", annotation_position="top right")
    hist.add_vline(x=0, line_dash="dash", line_color="#46DCC0", annotation_text="真の母平均", annotation_position="top left")
    _chart(hist, "再標本の平均（標準化単位）", "bootstrap平均の個数", 300)
    st.write("t区間は標本分散とt分布を使います。percentile区間はbootstrap平均の分位点を使い、"
             "BCaはbias correction z₀とdelete-one jackknifeによる加速係数aで分位点を補正します。"
             "percentileとBCaには同じ999個の平均を渡しています。")
    st.info("再標本化しても元の観測情報は増えません。nは観測数、B_boot=999は同じ原標本からの内側再標本化回数です。"
            "999を独立な母集団標本数とは数えません。有限B_bootの手続きであり、補正が常に被覆を改善するとは限りません。")


def _load_summary():
    path = EVIDENCE_DIR / "summary.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fixed_view(state):
    config = state["applied"]
    st.subheader("3. 保存済みの対応反復：同じ標本IDの3区間")
    st.caption("固定実験はα=0.05、真のμ=0・σ=1・d=0、各セルの独立な外側反復R=2,000、"
               "各標本の内側B_boot=999です。上の自由操作のα・試行IDとは独立した保存済み結果です。"
               "この画面では外側反復を再実行しません。")
    if config["alpha"] != 0.05:
        st.info(f"自由操作の適用済みα={config['alpha']:g}に対し、以下は固定α=0.05の結果です。信頼水準が異なります。")
    rows = [row for row in _load_summary()
            if row["distribution"] == config["dist"] and int(row["n"]) == config["n"]]
    if not rows:
        st.info("この分布・nの固定実験結果はまだ保存されていません。自由操作の1標本で代用せず、再実験も起動しません。")
        return
    present = {(int(row["master_seed"]), row["method"]) for row in rows}
    missing = [(seed, method) for seed in (0, 1, 2) for method in METHODS if (seed, method) not in present]
    if missing:
        st.warning("固定比較は未完了です。未保存のseed・手法：" + "、".join(
            f"seed={seed}/{METHODS[method]}" for seed, method in missing))
    st.session_state.setdefault("boot_fixed_seed", state["fixed_seed"])
    seed = st.selectbox("保存区間図のmaster seed", [0, 1, 2], key="boot_fixed_seed", on_change=_remember_fixed_seed)
    path = EVIDENCE_DIR / "intervals" / f"{config['dist']}_n{config['n']}_seed{seed}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["replicate_id"][:50]
            stored_methods = [str(value) for value in saved["methods"]]
            lower, upper = saved["lower"][:, :50], saved["upper"][:, :50]
            valid = saved["valid"][:, :50]
            reasons = saved["reason"][:, :50]
            warning = saved["tail_warning"][:, :50]
        fig = go.Figure()
        method_offsets = {"t": -0.24, "percentile": 0.0, "bca": 0.24}
        for method in METHODS:
            if method not in state["methods"]:
                continue
            row = stored_methods.index(method)
            xs, ys, text = [], [], []
            for index, replicate_id in enumerate(ids):
                if not valid[row, index]:
                    continue
                xs.extend([lower[row, index], upper[row, index], None])
                ys.extend([float(replicate_id) + method_offsets[method]] * 2 + [None])
                label = f"replicate ID={replicate_id} ／ {METHODS[method]}"
                text.extend([label, label, None])
            fig.add_scatter(x=xs, y=ys, text=text, mode="lines", name=METHODS[method],
                            line=dict(color=COLORS[method], dash=DASHES[method]),
                            hovertemplate="%{text}<br>端点=%{x:.5g}<extra></extra>")
        fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="真の母平均 μ=0")
        fig.update_yaxes(autorange="reversed")
        _chart(fig, "固定95%信頼区間（標準化単位）", "共通の原標本replicate ID（方法ごとに微小に上下配置）", 570)
        selected_rows = [stored_methods.index(method) for method in METHODS if method in state["methods"]]
        invalid_count = int(np.count_nonzero(~valid[selected_rows]))
        warning_count = int(np.count_nonzero(warning[selected_rows]))
        st.caption(f"同じ先頭{ids.size}標本の区間を表示しています。表示範囲の無効区間{invalid_count}本は描画を省略し、"
                   f"尾部解像度警告は{warning_count}本です。集計は図の50標本ではなく、保存済み全Rから計算しています。")
        if invalid_count:
            st.warning("表示区間の無効理由：" + " ／ ".join(sorted(set(str(value) for value in reasons[~valid]))))
    else:
        st.warning("保存された区間端点ファイルが見つかりません。集計値だけを表示します。")
    st.subheader("全Rによる被覆率・幅・外した側：3seed全件")
    display_rows = []
    fields = ["master_seed", "method", "R", "B_boot", "alpha", "total", "valid_count", "invalid_count", "cover_count",
              "coverage", "coverage_low", "coverage_high", "coverage_mcse",
              "miss_above_count", "miss_above", "miss_above_low", "miss_above_high", "miss_above_mcse",
              "miss_below_count", "miss_below", "miss_below_low", "miss_below_high", "miss_below_mcse",
              "mean_width", "width_sd", "width_median", "width_mcse", "tail_warning_count"]
    for row in sorted(rows, key=lambda value: (int(value["master_seed"]), list(METHODS).index(value["method"]))):
        entry = {}
        for field in fields:
            value = row.get(field, "")
            if field == "method":
                entry[field] = METHODS[value]
            else:
                try:
                    number = float(value)
                    entry[field] = number if np.isfinite(number) else None
                except ValueError:
                    entry[field] = None
        display_rows.append(entry)
    st.dataframe(display_rows, hide_index=True, width="stretch")
    st.caption("coverageは真のμ=0を含む率、miss_aboveはL>μ（区間全体が真値より上）、"
               "miss_belowはU<μ（区間全体が真値より下）です。coverage_low/highは記述用95% Wilson区間、"
               "各mcseは外側反復によるMonte Carlo標準誤差です。B_boot→∞の内側誤差評価ではありません。"
               "無効があるセルの主集計は未定義です。無効を被覆失敗として数えたり、黙って分母から除いたりしません。")
    if any(int(row["invalid_count"]) > 0 for row in rows):
        st.warning("固定比較に無効結果を含むセルがあります。total・valid_count・invalid_countと科学レポートを確認してください。")
    st.caption("参照：evidence/rq_m02/summary.csv と intervals/*.npz。3seedの各行を保持し、都合のよいseedだけを選んだ集計にしていません。")


def render_bootstrap_comparison():
    st.header("Inference Lab · 信頼区間の方法比較")
    st.write("同じ原標本から、t・percentile bootstrap・BCaがどんな区間を作るか。被覆・幅・外した側を合わせて観察する。")
    st.caption("自由操作は真のμ=0、σ=1、d=0、B_boot=999で固定。正規N(0,1)と中心化指数Exp(scale=1)−1を比較します。")
    state = _state()
    for name, value in state["pending"].items():
        st.session_state.setdefault(f"boot_{name}", value)
    st.session_state.setdefault("boot_methods", state["methods"])
    st.sidebar.selectbox("比較する母集団", list(DISTRIBUTIONS), format_func=DISTRIBUTIONS.get,
                         key="boot_dist", on_change=_remember_pending)
    st.sidebar.selectbox("比較標本数 n", [5, 20, 100], key="boot_n", on_change=_remember_pending)
    st.sidebar.selectbox("比較の有意水準 α", [0.01, 0.05, 0.10], key="boot_alpha", on_change=_remember_pending)
    st.sidebar.number_input("比較のmaster seed", 0, 2**32 - 1, step=1, key="boot_master_seed", on_change=_remember_pending)
    st.sidebar.number_input("比較の試行ID", 0, 2**32 - 1, step=1, key="boot_replicate_id", on_change=_remember_pending)
    st.sidebar.multiselect("表示する区間手法", list(METHODS), format_func=METHODS.get,
                           key="boot_methods", on_change=_remember_methods)
    if st.sidebar.button("比較条件を適用して実行", key="boot_run", width="stretch"):
        _execute()
    st.sidebar.button("同じ適用条件で次の試行", key="boot_new", on_click=_new_trial,
                      disabled=state["applied"] is None or state["applied"]["replicate_id"] >= 2**32 - 1, width="stretch")
    st.sidebar.caption("α・手法表示・Lab/モードの往復では原標本とbootstrap平均を保持します。"
                       "再生成条件は実行ボタンで確定し、次の試行ボタンは適用済み条件の試行IDだけを増やします。")
    if state["applied"] is None:
        st.info("条件を設定し、「比較条件を適用して実行」を押してください。保存済み固定実験の再実行は行いません。")
        return
    if state["pending"] != state["applied"]:
        st.warning("入力した比較条件は未適用です。以下は直前の適用済み条件の結果です。")
    st.success("適用済み比較条件：" + _conditions(state["applied"]))
    if not state["methods"]:
        st.info("区間手法を表示していません。原標本とbootstrap平均は保持されています。")
    _sample_view(state)
    _fixed_view(state)
    st.subheader("読み方と限界")
    st.write("95%信頼区間は、同じ手続きを繰り返したときに真の母平均を含む割合についての性質です。"
             "実現した一つの区間に真値が95%の確率で存在するという意味ではありません。"
             "bootstrapの近似精度は元標本の情報、母分布、nと有限B_bootに依存します。小標本の不足を補正だけで解消できるとは限りません。")
