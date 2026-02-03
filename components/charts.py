"""
components/charts.py

Standardized charting utilities for:
- Exposure concentration
- Regional distribution
- Credit utilization
- Stress testing impact

All charts are optimized for executive and governance presentations.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -------------------------------------------------
# Helper: Consistent Theme
# -------------------------------------------------

def _base_layout(fig, title: str, yaxis_title: str = None):
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=13),
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    if yaxis_title:
        fig.update_yaxes(title=yaxis_title)

    fig.update_xaxes(title=None)

    return fig


# -------------------------------------------------
# Counterparty Concentration
# -------------------------------------------------

def plot_concentration(df: pd.DataFrame):
    """
    Bar chart showing top counterparty exposure.
    """

    fig = px.bar(
        df,
        x="counterparty_name",
        y="ceded_exposure_usd",
        text_auto=".2s",
        color="ceded_exposure_usd",
        color_continuous_scale="Blues",
    )

    fig = _base_layout(
        fig,
        title="Top Counterparty Exposure Concentration",
        yaxis_title="Ceded Exposure (USD)",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Exposure: $%{y:,.0f}<extra></extra>"
        )
    )

    return fig


# -------------------------------------------------
# Exposure by Region (Heatmap)
# -------------------------------------------------

def plot_exposure_by_region(heat_df: pd.DataFrame):
    """
    Heatmap showing exposure distribution by counterparty and region.
    """

    fig = px.imshow(
        heat_df,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(color="Exposure (USD)"),
    )

    fig = _base_layout(
        fig,
        title="Exposure Distribution: Counterparty × Region",
    )

    fig.update_traces(
        hovertemplate="Exposure: $%{z:,.0f}<extra></extra>"
    )

    return fig


# -------------------------------------------------
# Credit Utilization
# -------------------------------------------------

def plot_utilization(df: pd.DataFrame):
    """
    Bar chart showing utilization vs credit limits.
    """

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["counterparty_name"],
        y=df["total_ceded_exposure_usd"],
        name="Exposure",
        marker_color="#4C78A8",
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=df["counterparty_name"],
        y=df["credit_limit_usd"],
        name="Credit Limit",
        marker_color="#E45756",
        opacity=0.6,
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ))

    fig = _base_layout(
        fig,
        title="Counterparty Exposure vs Credit Limits",
        yaxis_title="USD",
    )

    fig.update_layout(barmode="group")

    # Risk thresholds
    fig.add_hline(
        y=0.8 * df["credit_limit_usd"].max(),
        line_dash="dash",
        line_color="orange",
        annotation_text="80% Watch Level",
        annotation_position="top left",
    )

    fig.add_hline(
        y=df["credit_limit_usd"].max(),
        line_dash="dash",
        line_color="red",
        annotation_text="Limit Threshold",
        annotation_position="top left",
    )

    return fig


# -------------------------------------------------
# Stress Impact
# -------------------------------------------------

def plot_stress_impact(df: pd.DataFrame):
    """
    Comparison of baseline vs stressed utilization.
    """

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["counterparty_name"],
        y=df["utilization_pct"],
        name="Base Utilization",
        marker_color="#72B7B2",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=df["counterparty_name"],
        y=df["stressed_utilization_pct"],
        name="Stressed Utilization",
        marker_color="#F58518",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))

    fig = _base_layout(
        fig,
        title="Stress Scenario: Credit Utilization Impact",
        yaxis_title="Utilization (%)",
    )

    fig.update_layout(barmode="group")

    # Breach line
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="red",
        annotation_text="Limit Breach",
        annotation_position="top left",
    )

    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="orange",
        annotation_text="Watch Level",
        annotation_position="top left",
    )

    return fig
