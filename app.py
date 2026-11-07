"""MATHSCAPE's local, offline-capable Streamlit interface."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mathscape import __version__
from mathscape.chaos import bifurcation, sensitivity
from mathscape.fractal import mandelbrot
from mathscape.probability import (
    beta_moments, gamma_moments, normal_moments, probability_curve,
)

st.set_page_config(page_title="MATHSCAPE｜数学を触れる実験室", page_icon="∿", layout="wide")

ACCENT = "#46DCC0"
SECOND = "#FFB86B"
LABS = ["Probability Explorer", "Chaos Explorer", "Fractal Explorer"]


def chart(fig, x_title, y_title, height=410):
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=35, r=25, t=25, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=x_title, yaxis_title=y_title,
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


@st.cache_data(max_entries=4, show_spinner=False)
def cached_bifurcation():
    return bifurcation()


@st.cache_data(max_entries=4, show_spinner=False)
def cached_fractal(real, imag, width, resolution, iterations):
    return mandelbrot(real, imag, width, resolution, iterations)


def probability_lab():
    st.header("Probability Explorer")
    st.write("密度のかたちを動かして、ばらつきと偏りを観察する。")
    distribution = st.sidebar.selectbox("分布", ["正規分布", "ガンマ分布", "ベータ分布"])
    if distribution == "正規分布":
        mu = st.sidebar.slider("平均 μ", -5.0, 5.0, 0.0, 0.1)
        sigma = st.sidebar.slider("標準偏差 σ", 0.2, 3.0, 1.0, 0.1)
        name, params = "normal", dict(mu=mu, sigma=sigma)
        mean, variance = normal_moments(**params)
        st.latex(r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),\quad x\in\mathbb{R},\ \sigma>0")
        meaning = "μ は分布の中心、σ は広がりです。σを大きくすると、総面積を保ちながら山が低く広がります。"
        limit = "定義域は実数全体です。描画は0.05%〜99.95%分位点の有限区間で、両端の裾を省略しています。"
        hint = "μを固定してσを0.2から3へ動かし、ピークの高さと分散の変化を比べてみましょう。"
    elif distribution == "ガンマ分布":
        k = st.sidebar.slider("shape k", 0.5, 8.0, 2.0, 0.1)
        theta = st.sidebar.slider("scale θ", 0.2, 5.0, 1.0, 0.1)
        name, params = "gamma", dict(k=k, theta=theta)
        mean, variance = gamma_moments(**params)
        st.latex(r"f(x)=\frac{x^{k-1}e^{-x/\theta}}{\Gamma(k)\theta^k},\quad x>0,\ k,\theta>0")
        meaning = "k は形状、θ は尺度（scale）です。θはrateではありません。平均はkθ、分散はkθ²。k=1では指数分布になります。"
        limit = "定義域はx>0です。x=0は描画格子から除外し、上側は99.9%分位点までです。k<1ではx→0+で密度が無限大に発散します。端点付近のピークを有限の格子だけで表し切れません。"
        hint = "kを0.5、1、3へ変え、原点付近の密度と山の位置を観察してみましょう。"
    else:
        alpha = st.sidebar.slider("形状 α", 0.5, 10.0, 2.0, 0.1)
        beta = st.sidebar.slider("形状 β", 0.5, 10.0, 2.0, 0.1)
        name, params = "beta", dict(alpha=alpha, beta=beta)
        mean, variance = beta_moments(**params)
        st.latex(r"f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\quad 0<x<1,\ \alpha,\beta>0")
        meaning = "αとβは形状を決めます。α=βなら左右対称、α=β=1なら一様分布です。平均はα/(α+β)、分散はαβ/[(α+β)²(α+β+1)]。"
        limit = "定義域は0<x<1です。x=0,1は描画格子から除外します。α<1ではx→0+、β<1ではx→1−で密度が無限大に発散します。省略した端点の密度をゼロとして扱っていません。"
        hint = "α=β=0.5、1、5を比べると、両端に偏る形から中央に集まる形への変化が見えます。"
    x, y = probability_curve(name, **params)
    if not np.all(np.isfinite(y)):
        raise ValueError("表示格子に非有限の密度があります。パラメータを見直してください。")
    a, b = st.columns(2)
    a.metric("平均 E[X]", f"{mean:.5g}")
    b.metric("分散 Var[X]", f"{variance:.5g}")
    chart(go.Figure(go.Scatter(x=x, y=y, name="確率密度 PDF", line=dict(color=ACCENT, width=3))), "x", "確率密度 f(x)")
    st.info(limit)
    st.caption("縦軸は表示格子上の有限値に合わせた自動範囲です。密度そのものを切り詰めていません。密度の高さは確率ではなく、区間の面積が確率を表します。")
    st.subheader("数学的な意味")
    st.write(meaning)
    st.subheader("試してみる")
    st.write(hint)
    st.caption("曲線は有限点を線で結んだ近似です。表示区間や端点付近の省略により、描画曲線の面積は必ずしも1になりません。")


def set_chaos_preset():
    presets = {"安定固定点 · r=2.5": 2.5, "周期2 · r=3.2": 3.2, "複雑な軌道 · r=3.9": 3.9}
    choice = st.session_state.chaos_preset
    if choice in presets:
        st.session_state.chaos_r = presets[choice]


def chaos_lab():
    st.header("Chaos Explorer")
    st.write("単純な決定論的ルールから生まれる、多様な時間変化。")
    st.latex(r"x_{n+1}=r x_n(1-x_n),\quad 0\leq r\leq4,\quad 0\leq x_0\leq1")
    st.session_state.setdefault("chaos_r", 3.8)
    st.sidebar.selectbox("観察プリセット", ["自由探索", "安定固定点 · r=2.5", "周期2 · r=3.2", "複雑な軌道 · r=3.9"], key="chaos_preset", on_change=set_chaos_preset)
    r = st.sidebar.slider("成長パラメータ r", 0.0, 4.0, step=0.01, key="chaos_r")
    x0 = st.sidebar.slider("初期値 x₀", 0.0, 1.0, 0.2, 0.001, format="%.3f")
    n = st.sidebar.slider("反復回数 N", 20, 400, 80, 10)
    exponent = st.sidebar.slider("初期差の指数 log₁₀(δ)", -12, -3, -8)
    first, second, difference = sensitivity(r, x0, n, 10.0**exponent)
    a, b, c = st.columns(3)
    a.metric("r", f"{r:.2f}")
    b.metric("x₀", f"{first[0]:.6g}")
    c.metric("実際の初期差", f"{difference[0]:.2e}")
    st.subheader("時系列と初期値感度")
    steps = np.arange(n + 1)
    fig = go.Figure()
    fig.add_scatter(x=steps, y=first, name="軌道 x", mode="lines+markers", marker_size=3, line_color=ACCENT)
    fig.add_scatter(x=steps, y=second, name="微小に異なる軌道 x′", line=dict(color=SECOND, dash="dot"))
    fig.update_yaxes(range=[0, 1])
    chart(fig, "反復 n（初期値 n=0 を含む）", "xₙ")
    chart(go.Figure(go.Scatter(x=steps, y=difference, name="絶対差", line_color=SECOND)), "反復 n", "|xₙ − x′ₙ|", height=230)
    st.caption(f"x₀′={second[0]:.15g}。通常はδを加算し、1を超える場合は減算して[0,1]内に保ちます。差が0になった場合も、そのまま表示します。")
    st.subheader("分岐図")
    rs, values = cached_bifurcation()
    fig = go.Figure(go.Scattergl(x=np.tile(rs, values.shape[0]), y=values.ravel(), mode="markers", marker=dict(size=1.2, color=ACCENT, opacity=0.45), name="過渡後の観測値"))
    fig.add_vline(x=r, line_color=SECOND, annotation_text="現在の r")
    fig.update_xaxes(range=[2.5, 4.0])
    fig.update_yaxes(range=[0, 1])
    chart(fig, "r", "過渡後の xₙ")
    st.info("分岐図はr=2.5〜4.0の400点で、各点の初期値0.2から400反復を捨て、続く80反復を描く数値近似です。サイドバーのx₀・Nとは独立した条件で、厳密な分岐集合を描いたものではありません。現在のrが区間外なら縦線は見えません。")
    st.subheader("数学的な意味・試してみる")
    st.write("初期値0.2では、r=2.5で固定点0.6への収束、r=3.2で周期2の振る舞いを観察できます。初期値0や1では例外的な軌道になります。プリセットはrのみを変更します。r=3.9で初期差を変え、2軌道が離れる様子を比べてみましょう。")
    st.write("この写像は、同じ初期条件から同じ軌道が得られる決定論的モデルです。見た目の不規則さは乱数とは異なります。有限時間での軌道差だけでは、カオスの厳密な証明になりません。浮動小数点の丸め誤差も長期の軌道に影響します。")


def move_fractal(action):
    state = st.session_state
    if action == "reset":
        state.f_real, state.f_imag, state.f_width = -0.7, 0.0, 3.0
    elif action in ("in", "out"):
        state.f_width = float(np.clip(state.f_width * (0.5 if action == "in" else 2.0), 1e-6, 6.0))
    else:
        dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}[action]
        state.f_real = float(np.clip(state.f_real + dx * state.f_width * 0.25, -4, 4))
        state.f_imag = float(np.clip(state.f_imag + dy * state.f_width * 0.25, -4, 4))


def fractal_lab():
    st.header("Fractal Explorer")
    st.write("複素平面を旅して、マンデルブロ集合の境界を探す。")
    st.latex(r"z_0=0,\quad z_{n+1}=z_n^2+c,\quad c\in\mathbb{C}")
    # Session State owns these values, including updates from button callbacks.
    st.session_state.setdefault("f_real", -0.7)
    st.session_state.setdefault("f_imag", 0.0)
    st.session_state.setdefault("f_width", 3.0)
    real = st.sidebar.number_input("中心 Re(c)", -4.0, 4.0, step=0.01, format="%.7f", key="f_real")
    imag = st.sidebar.number_input("中心 Im(c)", -4.0, 4.0, step=0.01, format="%.7f", key="f_imag")
    width = st.sidebar.number_input("表示幅（実軸・虚軸共通）", 1e-6, 6.0, step=0.05, format="%.7f", key="f_width")
    resolution = st.sidebar.slider("解像度（一辺の画素数）", 120, 450, 360, 30)
    iterations = st.sidebar.slider("最大反復回数", 20, 150, 80, 10)
    cols = st.sidebar.columns(2)
    cols[0].button("＋ ズームイン", on_click=move_fractal, args=("in",), width="stretch")
    cols[1].button("− ズームアウト", on_click=move_fractal, args=("out",), width="stretch")
    cols = st.sidebar.columns(4)
    for col, text, action in zip(cols, ["←", "↑", "↓", "→"], ["left", "up", "down", "right"]):
        col.button(text, on_click=move_fractal, args=(action,), width="stretch")
    st.sidebar.button("全体表示に戻す", on_click=move_fractal, args=("reset",), width="stretch")
    with st.spinner("複素平面を計算しています…"):
        x, y, counts = cached_fractal(real, imag, width, resolution, iterations)
    a, b, c = st.columns(3)
    a.metric("表示幅", f"{width:.7g}")
    b.metric("画素数", f"{resolution} × {resolution}")
    c.metric("発散未確認の画素", f"{np.count_nonzero(counts == 0) / counts.size:.1%}")
    fig = go.Figure(go.Heatmap(x=x, y=y, z=np.where(counts > 0, counts, np.nan), zmin=1, zmax=iterations, colorscale="Viridis", colorbar=dict(title="発散反復 n"), hovertemplate="Re=%{x:.7f}<br>Im=%{y:.7f}<br>初回 |zₙ|>2: n=%{z:.0f}<extra></extra>"))
    fig.add_trace(go.Heatmap(x=x, y=y, z=np.where(counts == 0, 1.0, np.nan), colorscale=[[0, "#080C14"], [1, "#080C14"]], showscale=False, hovertemplate="Re=%{x:.7f}<br>Im=%{y:.7f}<br>最大反復まで発散未確認<extra></extra>"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    chart(fig, "実部 Re(c)", "虚部 Im(c)", height=620)
    st.caption("色付き：|zₙ|>2を初めて満たした反復回数。黒：最大反復まで発散未確認。実軸と虚軸は等縮尺です。")
    st.info("有限反復で発散が見つからなかった点は「非発散観測点」です。集合への所属が証明されたわけではありません。|zₙ|>2が観測された点は集合の外部と判定できます。画素間の細部も有限解像度では捉えられません。")
    st.subheader("数学的な意味・試してみる")
    st.write("マンデルブロ集合は、反復列が有界であり続ける複素数cの集合です。中心をRe(c)=−0.75、Im(c)=0.1付近に移し、ズームインして境界の細部を探してみましょう。最大反復数を増やしたとき、黒い領域の判定がどう変わるかも観察できます。")
    st.caption("再計算する範囲はサイドバーの中心・表示幅とボタンで変更します。Plotly上のドラッグズームは既存画像の表示操作で、再計算は行いません。解像度は最大450×450、反復数は最大150、表示幅は0.000001〜6に制限しています。")


st.title("MATHSCAPE")
st.write("数学を触れる実験室 · パラメータを動かして、数式の風景を探索する。")
st.sidebar.title("LABORATORY")
lab = st.sidebar.radio("Labを選択", LABS)
st.sidebar.divider()
try:
    {LABS[0]: probability_lab, LABS[1]: chaos_lab, LABS[2]: fractal_lab}[lab]()
except (ValueError, ArithmeticError) as error:
    st.error(f"計算条件を確認してください：{error}")
st.divider()
st.caption(f"MATHSCAPE v{__version__} · 有限精度の計算で数学を探索するローカル実験室")
