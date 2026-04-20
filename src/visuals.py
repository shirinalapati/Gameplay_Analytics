"""
Plotly chart builders — dark, operations-room styling for Streamlit.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Theme: slate base + high-contrast series (no orange / no Plasma yellow-orange) ---
BG = "#111827"
GRID = "#374151"
TEXT = "#f9fafb"
HOME_COLOR = "#34d399"  # emerald — readable on dark
AWAY_COLOR = "#60a5fa"  # blue

# Blue-only threat scale for shot map (WCAG-friendly on dark backgrounds)
THREAT_COLORSCALE = [
    [0.0, "#172554"],
    [0.25, "#1d4ed8"],
    [0.5, "#2563eb"],
    [0.75, "#38bdf8"],
    [1.0, "#e0f2fe"],
]


def _base_layout(title: str, height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=16)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        height=height,
        margin=dict(l=48, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )


def fig_cumulative_threat(
    shots: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    quality_col: str = "shot_quality",
    time_col: str = "game_seconds",
    current_time: Optional[float] = None,
) -> go.Figure:
    fig = go.Figure()
    if shots.empty:
        fig.update_layout(**_base_layout("Cumulative shot quality (threat index)", height=360))
        fig.add_annotation(
            text="No shot attempts in selection.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT, size=14),
        )
        if current_time is not None and current_time > 0:
            fig.add_vline(
                x=float(current_time),
                line_dash="dot",
                line_color="#93c5fd",
                line_width=1.2,
            )
            fig.update_xaxes(range=[0, max(1.0, float(current_time))])
        return fig

    has_trace = False
    for abbr, color in ((home_abbr, HOME_COLOR), (away_abbr, AWAY_COLOR)):
        sub = shots[shots["team_abbr"] == abbr].sort_values(time_col)
        if sub.empty:
            if current_time is None:
                continue
            x_vals = [0.0, float(current_time)]
            y_vals = [0.0, 0.0]
        else:
            cum = sub[quality_col].cumsum()
            x_vals = list(sub[time_col].astype(float))
            y_vals = list(cum.astype(float))
            if current_time is not None and float(current_time) > x_vals[-1]:
                x_vals.append(float(current_time))
                y_vals.append(float(y_vals[-1]))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=abbr,
                line=dict(width=2.4, color=color, shape="hv"),
            )
        )
        has_trace = True

    fig.update_layout(**_base_layout("Cumulative shot quality (threat index)", height=380))
    fig.update_xaxes(title="Game time (seconds)")
    fig.update_yaxes(title="Cumulative threat")
    if current_time is not None:
        ct = float(current_time)
        fig.add_vline(
            x=ct,
            line_dash="dot",
            line_color="#93c5fd",
            line_width=1.2,
        )
        max_t = ct
        if has_trace:
            xs = []
            for tr in fig.data:
                xs.extend(list(tr.x))
            if xs:
                max_t = max(max_t, float(max(xs)))
        fig.update_xaxes(range=[0, max(1.0, max_t)])
    return fig


def fig_rolling_momentum(
    shots: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    window_sec: float,
    quality_col: str = "shot_quality",
    time_col: str = "game_seconds",
) -> go.Figure:
    fig = go.Figure()
    if shots.empty or window_sec <= 0:
        fig.update_layout(**_base_layout(f"Rolling threat (last {int(window_sec)}s)", height=360))
        return fig

    times = np.sort(shots[time_col].unique())
    hx, hy = [], []
    ax, ay = [], []
    for t in times:
        w = shots[(shots[time_col] > t - window_sec) & (shots[time_col] <= t)]
        hx.append(t)
        hy.append(w[w["team_abbr"] == home_abbr][quality_col].sum())
        ax.append(t)
        ay.append(w[w["team_abbr"] == away_abbr][quality_col].sum())

    fig.add_trace(
        go.Scatter(x=hx, y=hy, mode="lines", name=home_abbr, line=dict(width=2, color=HOME_COLOR))
    )
    fig.add_trace(
        go.Scatter(x=ax, y=ay, mode="lines", name=away_abbr, line=dict(width=2, color=AWAY_COLOR))
    )
    fig.update_layout(**_base_layout(f"Rolling threat (last {int(window_sec)} seconds)", height=380))
    fig.update_xaxes(title="Game time (seconds)")
    fig.update_yaxes(title="Window threat sum")
    return fig


def fig_danger_counts(
    shots: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    time_col: str = "game_seconds",
) -> go.Figure:
    """Cumulative high-danger shot count by team."""
    fig = go.Figure()
    if shots.empty:
        fig.update_layout(**_base_layout("High-danger chances (cumulative)", height=320))
        return fig
    s = shots[shots["high_danger"] == 1].copy()
    for abbr, color in ((home_abbr, HOME_COLOR), (away_abbr, AWAY_COLOR)):
        sub = s[s["team_abbr"] == abbr].sort_values(time_col)
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[time_col],
                y=np.arange(1, len(sub) + 1),
                mode="lines+markers",
                name=abbr,
                line=dict(width=2, color=color),
                marker=dict(size=5),
            )
        )
    fig.update_layout(**_base_layout("High-danger chances (cumulative)", height=360))
    fig.update_xaxes(title="Game time (seconds)")
    fig.update_yaxes(title="Count")
    return fig


def fig_shot_map(
    shots: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    period_filter: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> go.Figure:
    fig = go.Figure()
    if shots.empty:
        fig.update_layout(**_base_layout("Shot map (threat-colored)", height=520))
        fig.add_annotation(
            text="No shots yet at this playback point. Press Play or scrub forward.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT, size=13),
        )
        return fig

    sub = shots.copy()
    if period_filter is not None:
        sub = sub[sub["period"] == period_filter]
    if max_seconds is not None:
        sub = sub[sub["game_seconds"] <= max_seconds]
    if sub.empty:
        fig.update_layout(**_base_layout("Shot map (threat-colored)", height=520))
        fig.add_annotation(text="No shots in this view.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    def _one_team(abbr: str, line_color: str, with_scale: bool) -> None:
        side = sub[sub["team_abbr"] == abbr]
        if side.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=side["x_coord"],
                y=side["y_coord"],
                mode="markers",
                name=abbr,
                marker=dict(
                    size=8 + 18 * side["shot_quality"].clip(0, 1),
                    color=side["shot_quality"],
                    colorscale=THREAT_COLORSCALE,
                    cmin=0,
                    cmax=1,
                    line=dict(width=0.6, color=line_color),
                    showscale=with_scale,
                    colorbar=dict(
                        title=dict(text="Threat", font=dict(color=TEXT, size=12)),
                        tickfont=dict(color=TEXT),
                    )
                    if with_scale
                    else None,
                ),
                text=side["shot_kind"],
                hovertemplate="%{text}<br>threat=%{marker.color:.2f}<br>x=%{x}<br>y=%{y}<extra></extra>",
            )
        )

    _one_team(home_abbr, HOME_COLOR, True)
    _one_team(away_abbr, AWAY_COLOR, False)

    fig.add_shape(
        type="rect",
        x0=-100,
        x1=100,
        y0=-42.5,
        y1=42.5,
        line=dict(color=GRID, width=1),
        fillcolor="rgba(255,255,255,0.02)",
    )
    fig.update_xaxes(range=[-100, 100], scaleanchor="y", title="X (ft, rink-fixed)")
    fig.update_yaxes(range=[-42.5, 42.5], title="Y (ft)")
    fig.update_layout(**_base_layout("Shot map (size/color = threat)", height=560))
    return fig


def fig_dual_bar_compare(
    labels: Sequence[str],
    home_vals: Sequence[float],
    away_vals: Sequence[float],
    home_abbr: str,
    away_abbr: str,
    title: str,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(name=home_abbr, x=list(labels), y=list(home_vals), marker_color=HOME_COLOR),
            go.Bar(name=away_abbr, x=list(labels), y=list(away_vals), marker_color=AWAY_COLOR),
        ]
    )
    fig.update_layout(
        barmode="group",
        **_base_layout(title, height=420),
    )
    fig.update_yaxes(title="Value")
    return fig


def fig_momentum_model(
    momentum_df: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    key_moments: Optional[dict] = None,
) -> go.Figure:
    """Plot explicit decayed momentum model (home/away/net) with key-moment markers."""
    fig = go.Figure()
    if momentum_df is None or momentum_df.empty:
        fig.update_layout(**_base_layout("Model-defined momentum", height=360))
        return fig
    m = momentum_df.sort_values("game_seconds")
    fig.add_trace(
        go.Scatter(
            x=m["game_seconds"],
            y=m["home_momentum"],
            mode="lines",
            name=f"{home_abbr} momentum",
            line=dict(color=HOME_COLOR, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=m["game_seconds"],
            y=m["away_momentum"],
            mode="lines",
            name=f"{away_abbr} momentum",
            line=dict(color=AWAY_COLOR, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=m["game_seconds"],
            y=m["net_momentum"],
            mode="lines",
            name="Net (home-away)",
            line=dict(color="#f8fafc", width=1.6, dash="dot"),
            opacity=0.8,
        )
    )
    if key_moments:
        swing = key_moments.get("swing")
        if swing:
            fig.add_vrect(
                x0=swing["start_sec"],
                x1=swing["end_sec"],
                fillcolor="rgba(56,189,248,0.16)",
                line_width=0,
                annotation_text=f"Biggest swing: {swing['team']}",
                annotation_position="top left",
            )
        tp = key_moments.get("turning_point")
        if tp:
            fig.add_vline(
                x=float(tp["sec"]),
                line_color="#fde047",
                line_width=1.5,
                line_dash="dash",
            )
    fig.update_layout(**_base_layout("Model-defined momentum (decayed + sequence weighted)", height=390))
    fig.update_xaxes(title="Game time (seconds)")
    fig.update_yaxes(title="Momentum score (unitless)")
    return fig
