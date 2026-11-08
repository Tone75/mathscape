"""Streamlit presentation and explicit experiment state for Inference Lab."""

import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

from mathscape.inference import generate_samples, normal_power, summarize, t_inference


ACCENT = "#46DCC0"
SECOND = "#FFB86B"
DISTRIBUTIONS = {"正規分布": "normal", "中心化指数分布（右に歪む）": "centered_exponential"}
DEFAULT = dict(distribution="normal", n=20, d=0.0, alpha=0.05, seed=0, repetitions=500)
PRESETS = {
    "差がなくても棄却することがある": DEFAULT,
    "同じ真の差でもnで検出しやすさが変わる": DEFAULT | {"d": 0.5},
    "母分布の歪み": DEFAULT | {"distribution": "centered_exponential", "n": 5},
}
OBSERVATIONS = {
    "差がなくても棄却することがある": "真の差d=0でも、反復すると棄却する標本があります。単一のp値ではなく全B回の第一種過誤率とMonte Carlo誤差を見てください。",
    "同じ真の差でもnで検出しやすさが変わる": "真の効果量dを固定し、n=5・20・100の固定実験を比べてください。一つの標本のp値や区間幅がnに対して単調になるとは限りません。",
    "母分布の歪み": "平均と母分散を正規分布と揃えた右に歪む母集団です。小標本の被覆率・第一種過誤率を観察し、ずれの向きや回復を決めつけずに比べてください。",
}
SUMMARY_PATH = Path(__file__).resolve().parents[1] / "evidence" / "rq_m01" / "summary.csv"
STATE_KEY = "inference_experiment"


def _state():
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {"pending": DEFAULT.copy(), "applied": None, "trial_id": 0, "fixed_seed": 0}
    # Preserve active experiments when a running development session reloads.
    st.session_state[STATE_KEY].setdefault("fixed_seed", 0)
    return st.session_state[STATE_KEY]


def _remember_pending():
    state = _state()
    for name in DEFAULT:
        key = f"inf_{name}"
        if key in st.session_state:
            state["pending"][name] = st.session_state[key]


def _preset():
    choice = st.session_state.inf_preset
    if choice in PRESETS:
        pending = PRESETS[choice].copy()
        _state()["pending"] = pending
        for name, value in pending.items():
            st.session_state[f"inf_{name}"] = value


def _remember_fixed_seed():
    _state()["fixed_seed"] = st.session_state.inf_fixed_seed


def _number(value, spec=".5g"):
    return format(value, spec) if np.isfinite(value) else "未定義"


def _execute(new_sample=False):
    state = _state()
    if new_sample:
        if state["applied"] is None:
            return
        config = state["applied"].copy()
        state["trial_id"] += 1
    else:
        config = state["pending"].copy()
    generation = {key: value for key, value in config.items() if key != "alpha"}
    generation["trial_id"] = state["trial_id"]
    # Alpha is deliberately excluded: inference changes, observations do not.
    if generation != state.get("generation"):
        samples = generate_samples(**generation)
    else:
        samples = state["samples"]
    result = t_inference(samples, mu0=0.0, alpha=config["alpha"])
    summary = summarize(result, true_mean=config["d"])
    state.update(applied=config, generation=generation, samples=samples, result=result, summary=summary)


def _chart(fig, x_title, y_title, height=380):
    fig.update_layout(template="plotly_dark", height=height,
                      margin=dict(l=35, r=25, t=35, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title=x_title, yaxis_title=y_title,
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def _conditions(config, trial_id):
    name = next(label for label, value in DISTRIBUTIONS.items() if value == config["distribution"])
    return (f"{name} ／ n={config['n']} ／ d={config['d']:g} ／ α={config['alpha']:g} ／ "
            f"seed={config['seed']} ／ B={config['repetitions']} ／ 試行ID={trial_id}")


def _single_sample(config, samples, result):
    st.subheader("1. 母集団と1回の標本")
    true_mean = config["d"]
    if config["distribution"] == "normal":
        st.latex(r"X=d+Z,\quad Z\sim N(0,1)")
        x = np.linspace(true_mean - 4, true_mean + 4, 500)
        density = stats.norm.pdf(x, loc=true_mean, scale=1)
    else:
        st.latex(r"X=d+E-1,\quad E\sim\mathrm{Exp}(\mathrm{scale}=1)")
        st.caption("中心化後のXを通常のガンマ分布とは呼びません。両母集団の真の平均はd、分散は1です。")
        x = np.linspace(true_mean - 1, true_mean + 7, 500)
        density = stats.expon.pdf(x, loc=true_mean - 1, scale=1)
    fig = go.Figure(go.Scatter(x=x, y=density, name="真の母集団PDF", line_color=ACCENT))
    fig.add_scatter(x=samples[0], y=np.zeros(config["n"]), mode="markers",
                    marker=dict(symbol="line-ns", size=15, color=SECOND), name="第1反復の標本点")
    fig.add_vline(x=true_mean, line_color=ACCENT, line_dash="dash", annotation_text="真の母平均 μ", annotation_position="top left")
    fig.add_vline(x=0, line_color=SECOND, line_dash="dot", annotation_text="帰無平均 μ₀=0", annotation_position="top right")
    _chart(fig, "X（標準化単位）", "確率密度／標本点は高さ0")
    st.caption("真の母平均と母標準偏差が分かるのは、この画面がシミュレーターだからです。真のσを推論計算へ渡さず、各標本からsを推定します。各反復内の観測と反復間を独立に生成し、欠測・外れ値除去・トリミングは行いません。PDFの表示は有限区間で裾を省略します。")
    cols = st.columns(3)
    cols[0].metric("真の効果量 d", f"{config['d']:.3g}")
    cols[1].metric("第1標本の平均", f"{result['mean'][0]:.5g}")
    cols[2].metric("第1標本の標準偏差 s", f"{result['sd'][0]:.5g}")
    if not result["valid"][0]:
        st.warning("第1標本のt推論は未定義です。s=0等の無効結果をp=0/1へ置き換えません。")
        return
    cols = st.columns(3)
    cols[0].metric("標準誤差 SE", f"{result['se'][0]:.5g}")
    cols[1].metric("検定統計量 T", f"{result['t'][0]:.5g}")
    cols[2].metric("両側p値", f"{result['p'][0]:.5g}")
    st.write(f"第1標本の{100 * (1 - config['alpha']):g}% t信頼区間："
             f"[{result['lower'][0]:.5g}, {result['upper'][0]:.5g}]。"
             f"判定：{'棄却（p < α）' if result['reject'][0] else '不棄却（p ≥ α）'}。"
             "表示丸め前のp値で判定します。")


def _repeated_intervals(config, result, summary):
    st.subheader("2. 実験を繰り返す：信頼区間と全B回の集計")
    limit = min(100, config["repetitions"])
    valid = result["valid"][:limit]
    lower, upper = result["lower"][:limit], result["upper"][:limit]
    covered = (lower <= config["d"]) & (config["d"] <= upper) & valid
    fig = go.Figure()
    for mask, label, color, dash in [
        (covered, "真の母平均を含む", ACCENT, "solid"),
        (valid & ~covered, "真の母平均を含まない", SECOND, "dash"),
    ]:
        xs, ys = [], []
        for index in np.flatnonzero(mask):
            xs.extend([lower[index], upper[index], None])
            ys.extend([index + 1, index + 1, None])
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=color, dash=dash), name=label)
    fig.add_vline(x=config["d"], line_color=ACCENT, line_dash="dash", annotation_text="真の母平均 μ", annotation_position="top left")
    fig.add_vline(x=0, line_color=SECOND, line_dash="dot", annotation_text="帰無平均 μ₀=0", annotation_position="top right")
    fig.update_yaxes(autorange="reversed")
    _chart(fig, "母平均のt信頼区間（標準化単位）", "反復番号（先頭から）", 450)
    st.caption(f"表示は先頭{limit}本だけです（この範囲の無効結果{int((~valid).sum())}本は線を描きません）。"
               f"以下の集計は全B={config['repetitions']}回に基づき、表示した区間だけで再集計しません。")
    rate_name = "第一種過誤率" if config["d"] == 0 else "検出力（棄却率）"
    cols = st.columns(3)
    cols[0].metric(rate_name, _number(summary["reject_rate"], ".2%"))
    cols[1].metric("真の母平均の被覆率", _number(summary["coverage"], ".2%"))
    cols[2].metric("平均信頼区間幅", _number(summary["mean_width"]))
    st.write(f"分母：{summary['denominator']} ／ 棄却数：{summary['reject_count']} ／ "
             f"被覆数：{summary['cover_count']} ／ 無効数：{summary['invalid_count']}。")
    st.dataframe([
        {"指標": rate_name, "推定値": summary["reject_rate"], "95% Wilson下限": summary["reject_low"],
         "95% Wilson上限": summary["reject_high"], "Monte Carlo標準誤差": summary["reject_mcse"]},
        {"指標": "被覆率", "推定値": summary["coverage"], "95% Wilson下限": summary["coverage_low"],
         "95% Wilson上限": summary["coverage_high"], "Monte Carlo標準誤差": summary["coverage_mcse"]},
    ], hide_index=True, width="stretch")
    st.caption(f"区間幅の標準偏差：{_number(summary['width_sd'])} ／ 平均幅のMonte Carlo標準誤差：{_number(summary['width_mcse'])}。"
               "Wilson区間は反復で得た率の二項区間です。各標本の母平均のt信頼区間とは別です。")
    if summary["invalid_count"]:
        st.warning("無効結果を含みます。分母と無効数を確認してください。無効を除いた都合のよい率として扱わず、当条件は要調査です。")
    if config["distribution"] == "normal":
        power = normal_power(config["n"], config["d"], config["alpha"])
        st.write(f"正規・独立同分布での非心t理論棄却率：{power:.2%}。有限反復の実測率にはMonte Carlo誤差があります。")
    else:
        st.info("中心化指数分布の結果は有限標本の実測です。正規母集団の非心t理論を厳密な理論検出力として流用しません。")


def _fixed_comparison(config):
    st.subheader("3. 標本数n=5・20・100による固定実験の比較")
    st.caption("ここは保存済みRQ-M01固定実験（各B=5,000、試行ID=0）です。"
               "上の自由探索のB・試行IDとは独立しています。同じ適用済み分布・d・αの行だけを照合します。")
    st.session_state.setdefault("inf_fixed_seed", _state()["fixed_seed"])
    master_seed = st.selectbox("固定比較のmaster seed", [0, 1, 2], key="inf_fixed_seed", on_change=_remember_fixed_seed)
    if config["alpha"] != 0.05 or not any(np.isclose(config["d"], d, atol=1e-12, rtol=0) for d in (0.0, 0.2, 0.5)):
        st.info("この適用条件に一致する固定結果はありません。固定実験はα=0.05、d=0・0.2・0.5です。近い条件の結果へ置き換えていません。")
        return
    if not SUMMARY_PATH.exists():
        st.info("固定実験ファイルはまだありません。自由探索の結果から固定実験を代用しません。")
        return
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["distribution"] == config["distribution"]
                and np.isclose(float(row["d"]), config["d"], atol=1e-12, rtol=0)
                and float(row["alpha"]) == config["alpha"]
                and int(row["master_seed"]) == master_seed]
    rows.sort(key=lambda row: int(row["n"]))
    if [int(row["n"]) for row in rows] != [5, 20, 100]:
        st.warning("一致する固定結果のn=5・20・100が揃っていません。ファイルを確認してください。")
        return
    ns = [int(row["n"]) for row in rows]
    fig = go.Figure()
    for key, low, high, label, color in [
        ("reject_rate", "reject_low", "reject_high", "第一種過誤率" if config["d"] == 0 else "実測検出力", SECOND),
        ("coverage", "coverage_low", "coverage_high", "実測被覆率", ACCENT),
    ]:
        values = np.array([float(row[key]) for row in rows])
        fig.add_scatter(x=ns, y=values, mode="lines+markers", name=label, line_color=color,
                        error_y=dict(type="data", symmetric=False,
                                     array=np.array([float(row[high]) for row in rows]) - values,
                                     arrayminus=values - np.array([float(row[low]) for row in rows])))
    if config["distribution"] == "normal":
        fig.add_scatter(x=ns, y=[float(row["theory_power"]) for row in rows], mode="lines+markers",
                        line=dict(dash="dot", color="white"), name="正規母集団の理論棄却率")
    fig.update_yaxes(range=[0, 1])
    _chart(fig, "1標本の観測数 n", "率（エラーバー：95% Wilson区間）")
    _chart(go.Figure(go.Scatter(x=ns, y=[float(row["mean_width"]) for row in rows],
                               mode="lines+markers", name="平均信頼区間幅", line_color=ACCENT)),
           "1標本の観測数 n", "平均信頼区間幅（標準化単位）", 280)
    fields = ["n", "B", "denominator", "invalid_count", "reject_rate", "coverage", "mean_width", "width_mcse"]
    st.dataframe([{key: float(row[key]) for key in fields} for row in rows], hide_index=True, width="stretch")
    st.caption(f"参照：evidence/rq_m01/summary.csv ／ master seed={master_seed}。"
               "seedは別の固定乱数系列の識別子であり、独立性の説明そのものではありません。")


def render_inference_lab():
    st.header("Inference Lab")
    st.write("母集団 → 標本 → 推定 → 反復した検定結果。標本数と分布の形で、推論の信頼性がどう変わるか。")
    st.latex(r"H_0:\mu=\mu_0,\ H_1:\mu\ne\mu_0,\quad T=\frac{\bar X-\mu_0}{s/\sqrt n},\quad \nu=n-1")
    st.latex(r"s^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar X)^2,\quad p=2\,\mathrm{sf}_{t_\nu}(|T|),\quad \bar X\pm t_{\nu,1-\alpha/2}\frac{s}{\sqrt n}")
    st.caption("μ₀=0、母標準偏差σ=1。真の母平均μ=d、真の効果量d=(μ−μ₀)/σ。"
               "本版は標準化単位で比較する、未知分散の1標本両側t推論です。")
    state = _state()
    for name, value in state["pending"].items():
        st.session_state.setdefault(f"inf_{name}", value)
    choice = st.sidebar.selectbox("推論の観察プリセット", ["自由探索", *PRESETS], key="inf_preset", on_change=_preset)
    st.sidebar.selectbox("母集団の分布", list(DISTRIBUTIONS.values()),
                         format_func=lambda value: next(k for k, v in DISTRIBUTIONS.items() if v == value),
                         key="inf_distribution", on_change=_remember_pending)
    st.sidebar.slider("1標本の観測数 n", 2, 200, key="inf_n", on_change=_remember_pending)
    st.sidebar.slider("真の標準化平均差 d", -1.0, 1.0, step=0.1, key="inf_d", on_change=_remember_pending)
    st.sidebar.selectbox("有意水準 α", [0.01, 0.05, 0.10], key="inf_alpha", on_change=_remember_pending)
    st.sidebar.number_input("master seed", 0, 2**32 - 1, step=1, key="inf_seed", on_change=_remember_pending)
    st.sidebar.number_input("独立実験の反復数 B", 1, 5000, step=100, key="inf_repetitions", on_change=_remember_pending)
    st.sidebar.caption("nは1回の標本数、Bは独立実験の反復数。n×B≤1,000,000。表示操作では標本を生成しません。")
    if st.sidebar.button("条件を適用して実行", key="inf_run", width="stretch"):
        _execute()
    if st.sidebar.button("適用条件で新しい標本", key="inf_new", disabled=state["applied"] is None, width="stretch"):
        _execute(new_sample=True)
    if choice in OBSERVATIONS:
        st.info(OBSERVATIONS[choice])
    if state["applied"] is None:
        st.info("サイドバーで条件を設定し、「条件を適用して実行」を押してください。プリセットも実行するまでは未適用です。")
        return
    config = state["applied"]
    if state["pending"] != config:
        st.warning("入力条件は未適用です。以下は直前に実行した適用済み条件の結果です。")
    st.success("適用済み条件：" + _conditions(config, state["generation"]["trial_id"]))
    _single_sample(config, state["samples"], state["result"])
    _repeated_intervals(config, state["result"], state["summary"])
    _fixed_comparison(config)
    st.subheader("読み方と限界")
    st.write("95%信頼区間とは、同じ手続きを繰り返したときに真の母平均を含む割合が95%になるという性質です。"
             "実現済みの一つの区間に真値が95%の確率で入る、という意味ではありません。正規・独立同分布ならt推論が厳密に成り立ち、非正規では有限標本の近似です。")
    st.write("p値は帰無仮説が正しい確率ではありません。不棄却は効果がないという証明ではなく、統計的有意性と効果の大きさ・実用上の重要性は別です。"
             "有意だから行動すべき、という自動判断は行いません。固定した真のd、推定値、区間と反復全体を合わせて読みます。")
