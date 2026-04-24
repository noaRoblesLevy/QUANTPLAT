from typing import Any, Dict, List

import plotly.graph_objects as go

_THEME = "plotly_dark"
_GREEN = "#00d4aa"
_RED = "#ff6b6b"
_LIGHT_GREEN = "#51cf66"


def equity_curve_chart(equity_curve: List[float], title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve,
        mode="lines",
        name="Equity",
        line=dict(color=_GREEN, width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Trade #",
        yaxis_title="Equity ($)",
        template=_THEME,
        height=400,
    )
    return fig


def monte_carlo_chart(mc_data: Dict[str, Any],
                      title: str = "Monte Carlo Simulation") -> go.Figure:
    p5 = mc_data["percentiles"]["p5"]
    p50 = mc_data["percentiles"]["p50"]
    p95 = mc_data["percentiles"]["p95"]
    x = list(range(len(p50)))
    x_fill = x + x[::-1]
    y_fill = p95 + p5[::-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_fill, y=y_fill,
        fill="toself",
        fillcolor="rgba(0,212,170,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="5th-95th percentile",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p95, mode="lines",
        name="95th percentile",
        line=dict(color=_LIGHT_GREEN, width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode="lines",
        name="Median",
        line=dict(color=_GREEN, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p5, mode="lines",
        name="5th percentile",
        line=dict(color=_RED, width=1, dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Trade #",
        yaxis_title="Equity ($)",
        template=_THEME,
        height=400,
    )
    return fig


def trade_distribution_chart(pl_list: List[float],
                              title: str = "Trade P&L Distribution") -> go.Figure:
    wins = [p for p in pl_list if p > 0]
    losses = [p for p in pl_list if p < 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=wins, name="Wins",
        marker_color=_LIGHT_GREEN, nbinsx=20, opacity=0.75,
    ))
    fig.add_trace(go.Histogram(
        x=losses, name="Losses",
        marker_color=_RED, nbinsx=20, opacity=0.75,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="P&L ($)",
        yaxis_title="Count",
        barmode="overlay",
        template=_THEME,
        height=350,
    )
    return fig
