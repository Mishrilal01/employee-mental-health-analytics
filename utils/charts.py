"""
charts.py
---------
Responsible for building every Plotly chart used in the dashboard.

Design rules followed throughout this module:
- Plotly Express / Graph Objects only (no static matplotlib).
- Every chart has a meaningful title and labeled axes.
- Counts and/or percentages are shown via data labels or hover text.
- Charts return `None` when there is no data, so callers can show a
  friendly "no data" message instead of rendering an empty figure.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Set2
ACCENT = "#1D8178"


def _empty_check(df: pd.DataFrame) -> bool:
    return df is None or df.empty


def apply_chart_layout(fig, height: int = 470, show_legend: bool = False):
    """Apply readable, responsive dimensions without changing chart meaning."""
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        autosize=True,
        margin=dict(t=96, b=92, l=78, r=160 if show_legend else 90),
        font=dict(family="DM Sans, sans-serif", color="#172331"),
        title_font=dict(family="Space Grotesk, sans-serif", size=16, color="#172331"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        showlegend=show_legend,
        title=dict(x=0.5, xanchor="center"),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if show_legend:
        fig.update_layout(
            legend=dict(
                orientation="v",
                y=1,
                x=1.02,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.86)",
                bordercolor="#DCE6E2",
                borderwidth=1,
            )
        )
    return fig


def category_bar_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    x_label: str | None = None,
    y_label: str = "Number of Respondents",
    horizontal: bool = False,
    order: list[str] | None = None,
    color: str | None = ACCENT,
):
    """Bar chart of value counts + percentage, for a single categorical column."""
    if _empty_check(df) or column not in df.columns:
        return None

    counts = df[column].value_counts(dropna=False)
    total = counts.sum()
    if total == 0:
        return None

    data = counts.rename_axis(column).reset_index(name="Count")
    data["Percentage"] = (data["Count"] / total * 100).round(2)

    if order:
        data[column] = pd.Categorical(data[column], categories=order, ordered=True)
        data = data.sort_values(column)
    else:
        data = data.sort_values("Count", ascending=horizontal)

    if horizontal:
        fig = px.bar(
            data, x="Count", y=column, orientation="h",
            title=title, text=data.apply(lambda r: f"{r['Count']:,} ({r['Percentage']}%)", axis=1),
            color_discrete_sequence=[color] if color else COLOR_SEQ,
        )
        fig.update_layout(xaxis_title=y_label, yaxis_title=x_label or column)
    else:
        fig = px.bar(
            data, x=column, y="Count",
            title=title, text=data.apply(lambda r: f"{r['Count']:,} ({r['Percentage']}%)", axis=1),
            color_discrete_sequence=[color] if color else COLOR_SEQ,
        )
        fig.update_layout(xaxis_title=x_label or column, yaxis_title=y_label)

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Count: %{y}<extra></extra>" if not horizontal else
                      "%{y}<br>Count: %{x}<extra></extra>",
    )
    if not horizontal:
        fig.update_xaxes(tickangle=-30)
    return apply_chart_layout(fig)


def age_histogram(df: pd.DataFrame, nbins: int = 20):
    """Histogram of respondent Age."""
    if _empty_check(df) or "Age" not in df.columns or df["Age"].dropna().empty:
        return None

    fig = px.histogram(
        df, x="Age", nbins=nbins,
        title="Age Distribution",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(xaxis_title="Age", yaxis_title="Number of Respondents", bargap=0.05)
    fig.update_traces(hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>")
    return apply_chart_layout(fig, height=470)


def treatment_rate_bar(
    rate_df: pd.DataFrame,
    group_col: str,
    title: str,
    x_label: str | None = None,
    order: list[str] | None = None,
):
    """Bar chart of Treatment Rate (%) by a single group column, with respondent counts on hover."""
    if _empty_check(rate_df):
        return None

    data = rate_df.dropna(subset=["Treatment Rate (%)"]).copy()
    if data.empty:
        return None

    if order:
        data[group_col] = pd.Categorical(data[group_col], categories=order, ordered=True)
        data = data.sort_values(group_col)
    else:
        data = data.sort_values("Treatment Rate (%)", ascending=False)

    fig = px.bar(
        data, x=group_col, y="Treatment Rate (%)",
        title=title,
        text=data["Treatment Rate (%)"].apply(lambda v: f"{v}%"),
        color_discrete_sequence=[ACCENT],
        custom_data=["Respondents"],
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"%{{x}}<br>Treatment Rate: %{{y}}%<br>Respondents: %{{customdata[0]}}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title=x_label or group_col,
        yaxis_title="Treatment Rate (%)",
        yaxis_range=[0, 100],
    )
    fig.update_xaxes(tickangle=-30)
    return apply_chart_layout(fig, height=480)


def grouped_treatment_rate_bar(
    rate_df: pd.DataFrame,
    group_col_1: str,
    group_col_2: str,
    title: str,
):
    """Grouped bar chart: Treatment Rate (%) by two categorical columns."""
    if _empty_check(rate_df):
        return None

    data = rate_df.dropna(subset=["Treatment Rate (%)"]).copy()
    if data.empty:
        return None

    fig = px.bar(
        data, x=group_col_1, y="Treatment Rate (%)", color=group_col_2,
        barmode="group", title=title,
        text=data["Treatment Rate (%)"].apply(lambda v: f"{v}%"),
        color_discrete_sequence=COLOR_SEQ,
        custom_data=["Respondents"],
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"%{{x}}<br>{group_col_2}: %{{fullData.name}}"
            "<br>Treatment Rate: %{y}%<br>Respondents: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title=group_col_1,
        yaxis_title="Treatment Rate (%)",
        yaxis_range=[0, 100],
        legend_title=group_col_2,
    )
    fig.update_xaxes(tickangle=-30)
    return apply_chart_layout(fig, height=540, show_legend=True)


def correlation_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Heatmap"):
    """Heatmap for a correlation matrix produced by calculations.correlation_matrix."""
    if corr_df is None or corr_df.empty:
        return None

    fig = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title,
        aspect="auto",
    )
    return apply_chart_layout(fig, height=600)


def treatment_pie(df: pd.DataFrame):
    """Donut chart of overall treatment distribution."""
    if _empty_check(df) or "treatment" not in df.columns:
        return None

    counts = df["treatment"].value_counts()
    if counts.empty:
        return None

    fig = px.pie(
        names=counts.index, values=counts.values,
        title="Treatment Distribution", hole=0.45,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
    )
    return apply_chart_layout(fig, height=470, show_legend=True)
