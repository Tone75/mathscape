"""Render saved RQ-M01 summaries with Plotly; never generate new samples."""

import argparse
import csv
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(summary, output):
    if output.exists():
        raise FileExistsError("Preserving existing figure; choose a new output path")
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 54:
        raise ValueError("Expected all 54 fixed cells")
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "d=0: coverage of true mean", "d=0: type I error",
        "d=0: mean t-interval width", "d=0.5: power (rejection rate)",
    ], vertical_spacing=0.18)
    for distribution, color in [("normal", "#007c83"), ("centered_exponential", "#a84624")]:
        for seed, symbol in [(0, "circle"), (1, "square"), (2, "triangle-up")]:
            for row, col, effect, metric, low, high in [
                (1, 1, 0.0, "coverage", "coverage_low", "coverage_high"),
                (1, 2, 0.0, "reject_rate", "reject_low", "reject_high"),
                (2, 1, 0.0, "mean_width", None, None),
                (2, 2, 0.5, "reject_rate", "reject_low", "reject_high"),
            ]:
                selected = sorted([r for r in rows if r["distribution"] == distribution
                                   and int(r["master_seed"]) == seed and float(r["d"]) == effect],
                                  key=lambda r: int(r["n"]))
                values = [float(r[metric]) for r in selected]
                error = (dict(type="data", symmetric=False,
                              array=[float(r[high]) - v for r, v in zip(selected, values)],
                              arrayminus=[v - float(r[low]) for r, v in zip(selected, values)])
                         if low else dict(type="data", array=[float(r["width_mcse"]) for r in selected]))
                fig.add_trace(go.Scatter(
                    x=[str(r["n"]) for r in selected], y=values,
                    mode="lines+markers", name=f"{distribution}, seed={seed}",
                    legendgroup=f"{distribution}-{seed}", showlegend=row == col == 1,
                    line=dict(color=color, dash="solid" if distribution == "normal" else "dash"),
                    marker=dict(symbol=symbol, size=7), error_y=error,
                ), row=row, col=col)
    theory = sorted([r for r in rows if r["distribution"] == "normal"
                     and int(r["master_seed"]) == 0 and float(r["d"]) == .5], key=lambda r: int(r["n"]))
    fig.add_trace(go.Scatter(x=[r["n"] for r in theory], y=[float(r["theory_power"]) for r in theory],
                            mode="lines", name="Normal theory (d=0.5)",
                            line=dict(color="#222222", dash="dot")), row=2, col=2)
    fig.add_hline(y=.95, line_dash="dot", row=1, col=1)
    fig.add_hline(y=.05, line_dash="dot", row=1, col=2)
    fig.update_xaxes(title_text="Observations per sample n", type="category")
    for row, col in [(1, 1), (1, 2), (2, 2)]:
        fig.update_yaxes(title_text="Rate; 95% Wilson intervals", row=row, col=col)
    fig.update_yaxes(title_text="Width; bars = MC standard error", row=2, col=1)
    fig.update_layout(template="plotly_white", width=1200, height=900,
                      title="RQ-M01: fixed results, all 3 seeds; alpha=0.05, B=5000 per cell",
                      legend=dict(orientation="h", y=-.18), margin=dict(b=160))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True, full_html=True,
                   config={"displaylogo": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("evidence/rq_m01/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("evidence/rq_m01/figures/overview.html"))
    args = parser.parse_args()
    render(args.summary, args.output)
