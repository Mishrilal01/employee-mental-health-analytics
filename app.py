"""
Employee Mental Health Analytics Dashboard
--------------------------------------------
Interactive Exploratory Data Analysis of Workplace Mental Health.

This is an EDA + dashboard project only. No machine learning model,
prediction, classifier, or model training is included anywhere in this
application.
"""

import pandas as pd
import streamlit as st
from html import escape

from utils.data_loader import (
    load_data,
    dataset_overview,
    DatasetError,
    AGE_GROUP_ORDER,
    WORK_INTERFERE_ORDER,
)
from utils.filters import render_sidebar_filters, apply_filters, filter_summary
from utils.calculations import (
    compute_kpis,
    treatment_rate_by,
    treatment_rate_by_two,
    top_countries,
    correlation_matrix,
    is_small_sample,
    SMALL_SAMPLE_THRESHOLD,
)
from utils import charts

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Mental Health Analytics Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# VISUAL SYSTEM
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root {--ink:#172331;--muted:#6f7c86;--paper:#f6f8f7;--panel:#ffffff;--teal:#1d8178;--teal-soft:#e2f2ef;--coral:#e27d62;--line:#dce6e2;}
        html, body, [class*="css"] {font-family:'DM Sans',sans-serif;color:var(--ink);}
        h1, h2, h3, h4 {font-family:'Space Grotesk',sans-serif;letter-spacing:0;color:var(--ink);}
        .stApp {background-color:var(--paper);background-image:radial-gradient(circle at 92% 4%,rgba(29,129,120,.11) 0,rgba(29,129,120,0) 24rem),linear-gradient(120deg,rgba(255,255,255,.55),rgba(246,248,247,.9));}
        [data-testid="stHeader"] {background:transparent;}
        .block-container {max-width:1480px;padding-top:2.4rem;padding-bottom:3rem;}
        [data-testid="stSidebar"] {background:#172331;border-right:1px solid #243849;}
        [data-testid="stSidebar"] * {color:#eaf2f0;}
        [data-testid="stSidebar"] [data-baseweb="select"] > div {background:#243849;border-color:#3b5364;}
        [data-testid="stSidebar"] hr {border-color:#385060;}
        [data-testid="stSidebar"] button {border-radius:8px;border:1px solid #587080;background:#243849;}
        [data-testid="stSidebar"] button:hover {border-color:#83d0c4;color:#fff;}
        .hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#172331 0%,#214b55 100%);border-radius:18px;padding:2.35rem 2.5rem 2.15rem;margin-bottom:1.5rem;box-shadow:0 14px 34px rgba(23,35,49,.14);}
        .hero:after {content:'✦';position:absolute;right:2.4rem;top:1rem;color:rgba(131,208,196,.7);font-size:8rem;line-height:1;transform:rotate(15deg);}
        .hero:before {content:'';position:absolute;right:-4rem;bottom:-5rem;width:17rem;height:17rem;border:1px solid rgba(131,208,196,.25);border-radius:50%;box-shadow:0 0 0 1.3rem rgba(131,208,196,.04),0 0 0 2.6rem rgba(131,208,196,.03);}
        .hero h1 {position:relative;z-index:1;color:#fff;font-size:2.65rem;margin:0;line-height:1.05;white-space:nowrap;}
        .hero p {position:relative;z-index:1;color:#b9d7d2;font-size:1.05rem;margin:.75rem 0 0;max-width:34rem;}
        .eyebrow {color:#83d0c4;text-transform:uppercase;font-size:.72rem;font-weight:700;letter-spacing:.12em;margin-bottom:.7rem;}
        [data-testid="stMetric"] {background:var(--panel);border:1px solid var(--line);border-top:4px solid var(--teal);border-radius:12px;padding:1rem 1.1rem .85rem;box-shadow:0 5px 16px rgba(23,35,49,.05);min-height:108px;}
        [data-testid="stMetricLabel"] {color:var(--muted);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.035em;white-space:normal;line-height:1.2;}
        [data-testid="stMetricValue"] {color:var(--ink);font-family:'Space Grotesk',sans-serif;font-size:clamp(1.45rem,2.2vw,2rem);white-space:nowrap;}
        [data-testid="stTabs"] [role="tablist"] {gap:.45rem;border:1px solid var(--line);border-radius:12px;background:rgba(255,255,255,.78);padding:.35rem .45rem;box-shadow:0 5px 16px rgba(23,35,49,.05);}
        [data-testid="stTabs"] button {color:var(--muted);font-weight:600;padding:.62rem .85rem;border-radius:8px;border-bottom:0;transition:background .2s ease,color .2s ease;}
        [data-testid="stTabs"] button:hover {color:var(--teal);background:var(--teal-soft);}
        [data-testid="stTabs"] button[aria-selected="true"] {color:#fff;background:var(--teal);border-bottom:0;box-shadow:0 3px 8px rgba(29,129,120,.22);}
        [data-testid="stExpander"] {border:1px solid var(--line);border-radius:10px;background:var(--panel);}
        [data-testid="stPlotlyChart"] {background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.35rem .45rem 0;box-shadow:0 5px 16px rgba(23,35,49,.04);}
            [data-testid="stPlotlyChart"] > div,
            [data-testid="stPlotlyChart"] [data-testid="stFullScreenFrame"],
            [data-testid="stPlotlyChart"] .js-plotly-plot,
            [data-testid="stPlotlyChart"] .plot-container,
            [data-testid="stPlotlyChart"] .svg-container {overflow:visible !important;}
            [data-testid="stElementContainer"]:has(> [data-testid="stFullScreenFrame"]),
            [data-testid="stFullScreenFrame"] {overflow:visible !important;}
            [data-testid="stPlotlyChart"] .modebar {z-index:20 !important;right:10px !important;top:6px !important;}
        [data-testid="stElementContainer"].st-key-treatment-overview,
        [data-testid="stElementContainer"].st-key-treatment-overview > *,
        [data-testid="stElementContainer"].st-key-treatment-overview [data-testid="stFullScreenFrame"],
        [data-testid="stElementContainer"].st-key-treatment-overview [data-testid="stPlotlyChart"],
        [data-testid="stElementContainer"].st-key-treatment-overview .js-plotly-plot,
        [data-testid="stElementContainer"].st-key-treatment-overview .plot-container,
        [data-testid="stElementContainer"].st-key-treatment-overview .svg-container {overflow:visible !important;}
        [data-testid="stColumn"]:has(.st-key-treatment-overview) {overflow:visible !important;position:relative;z-index:5;}
        [data-testid="stElementContainer"].st-key-treatment-overview .modebar {z-index:30 !important;right:10px !important;top:6px !important;}
        [data-testid="stElementContainer"].st-key-treatment-treatment,
        [data-testid="stElementContainer"].st-key-treatment-treatment > *,
        [data-testid="stElementContainer"].st-key-treatment-treatment [data-testid="stFullScreenFrame"],
        [data-testid="stElementContainer"].st-key-treatment-treatment [data-testid="stPlotlyChart"],
        [data-testid="stElementContainer"].st-key-treatment-treatment .js-plotly-plot,
        [data-testid="stElementContainer"].st-key-treatment-treatment .plot-container,
        [data-testid="stElementContainer"].st-key-treatment-treatment .svg-container {overflow:visible !important;}
        [data-testid="stColumn"]:has(.st-key-treatment-treatment) {overflow:visible !important;position:relative;z-index:5;}
        [data-testid="stElementContainer"].st-key-treatment-treatment .modebar {z-index:30 !important;right:10px !important;top:6px !important;}
        .filter-strip {background:linear-gradient(115deg,#e5f4f1 0%,#f8fbfa 72%);border:1px solid #c9e5df;border-radius:13px;padding:1rem 1.1rem 1.05rem;margin:.3rem 0 1.2rem;box-shadow:0 6px 18px rgba(29,129,120,.07);}
        .filter-strip p {margin:0;color:var(--muted);font-size:.72rem;}
        .filter-heading {display:flex;align-items:center;gap:.55rem;margin-bottom:.85rem;}
        .filter-heading strong {color:var(--teal);font-size:.72rem;letter-spacing:.12em;}
        .filter-heading span {height:1px;flex:1;background:#c9e5df;}
        .filter-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem .8rem;}
        .filter-item {background:rgba(255,255,255,.72);border:1px solid rgba(201,229,223,.9);border-radius:8px;padding:.55rem .7rem;min-width:0;}
        .filter-item b {display:block;color:var(--muted);font-size:.66rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:.14rem;}
        .filter-item span {display:block;color:var(--ink);font-size:.84rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
        .filter-total {margin-top:.75rem!important;color:var(--teal)!important;font-weight:700;font-size:.8rem!important;}
        @media (max-width: 900px) {.filter-grid {grid-template-columns:repeat(2,minmax(0,1fr));}}
        .caution-note {background:#fff4e6;border-left:4px solid var(--coral);color:#694133;padding:.7rem .95rem;border-radius:7px;font-size:.92rem;}
        .section-kicker {display:inline-block;color:var(--teal);background:var(--teal-soft);border-radius:999px;padding:.28rem .65rem;text-transform:uppercase;font-size:.68rem;font-weight:700;letter-spacing:.1em;margin-bottom:.15rem;}
        div[data-testid="stDownloadButton"] button {border-radius:8px;border:1px solid var(--teal);color:var(--teal);font-weight:700;}
        div[data-testid="stDownloadButton"] button:hover {background:var(--teal);color:#fff;}
        @media (max-width: 1200px) {
            [data-testid="stHorizontalBlock"] {flex-wrap:wrap;row-gap:1rem;}
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {min-width:100% !important;width:100% !important;max-width:100% !important;flex:1 1 100% !important;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# TITLE
# ----------------------------------------------------------------------
st.markdown(
    '<div class="hero"><div class="eyebrow">People analytics · exploratory view</div>'
    '<h1>Employee mental health</h1>'
    '<p>Understand the workplace signals behind treatment-seeking patterns.</p></div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# LOAD DATA (with graceful error handling)
# ----------------------------------------------------------------------
try:
    raw_df = load_data()
except DatasetError as e:
    st.error(f"⚠️ {e}")
    st.stop()

# ----------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------
selections = render_sidebar_filters(raw_df)
filtered_df: pd.DataFrame = apply_filters(raw_df, selections)

# ----------------------------------------------------------------------
# ABOUT / OBJECTIVE
# ----------------------------------------------------------------------
with st.expander("ℹ️ About / Business Objective", expanded=False):
    st.write(
        "To analyze employee mental-health survey data and identify demographic and "
        "workplace factors associated with seeking mental-health treatment, so that "
        "organizations can better understand gaps in mental-health support and make "
        "data-informed improvements to workplace well-being initiatives."
    )
    st.caption(
        "This dashboard presents exploratory, descriptive analytics only. It does not "
        "diagnose individuals, rank employees, or make predictions about any single person."
    )

# ----------------------------------------------------------------------
# EMPTY-FILTER GUARD
# ----------------------------------------------------------------------
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust the filters.")
    st.stop()

# ----------------------------------------------------------------------
# CURRENT FILTER SUMMARY
# ----------------------------------------------------------------------
with st.container():
    summary = filter_summary(selections)
    filter_items = "".join(
        f'<div class="filter-item"><b>{escape(str(label))}</b>'
        f'<span>{escape(str(value))}</span></div>'
        for label, value in summary.items()
    )
    st.markdown(
        '<div class="filter-strip">'
        '<div class="filter-heading"><strong>ACTIVE VIEW</strong><span></span></div>'
        f'<div class="filter-grid">{filter_items}</div>'
        f'<p class="filter-total">Filtered respondents · {len(filtered_df):,} of {len(raw_df):,}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ----------------------------------------------------------------------
# TOP KPI CARDS
# ----------------------------------------------------------------------
kpis = compute_kpis(filtered_df)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Respondents", f"{kpis['total_respondents']:,}")
k2.metric("Treatment Rate", f"{kpis['treatment_rate']:.2f}%" if pd.notna(kpis["treatment_rate"]) else "N/A")
k3.metric("Average Age", f"{kpis['average_age']:.2f}" if pd.notna(kpis["average_age"]) else "N/A")
k4.metric("Remote Work Rate", f"{kpis['remote_work_rate']:.2f}%" if pd.notna(kpis["remote_work_rate"]) else "N/A")
k5.metric("Family History Rate", f"{kpis['family_history_rate']:.2f}%" if pd.notna(kpis["family_history_rate"]) else "N/A")

st.markdown("---")


def small_sample_note(n: int, context: str = "this group"):
    if is_small_sample(n, SMALL_SAMPLE_THRESHOLD):
        st.markdown(
            f'<div class="caution-note">⚠️ Note: {context} has a small sample size '
            f'(n = {n}). Percentages should be interpreted cautiously.</div>',
            unsafe_allow_html=True,
        )


_chart_counter = {"n": 0}


def show_chart_or_message(fig, empty_message="No data available for this chart with the current filters.", key=None):
    """Render a Plotly figure, or a friendly message when there is no data.

    Every call gets a unique Streamlit element key (auto-incremented if not
    provided explicitly) so the same chart type/data can safely appear more
    than once across different tabs without an ID collision.
    """
    if fig is None:
        st.info(empty_message)
        return
    if key is None:
        _chart_counter["n"] += 1
        key = f"chart_{_chart_counter['n']}"
    st.plotly_chart(
        fig,
        width="stretch",
        config={"responsive": True},
        key=key,
    )


# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tabs = st.tabs([
    "Overview",
    "Demographics",
    "Mental Health & Treatment",
    "Workplace Factors",
    "Support & Resources",
    "Multivariate Analysis",
    "Key Findings",
])

# ====================== A. OVERVIEW ======================
with tabs[0]:
    st.markdown('<div class="section-kicker">At a glance</div>', unsafe_allow_html=True)
    st.subheader("Overview")

    if filtered_df["Gender"].value_counts().min() if not filtered_df["Gender"].value_counts().empty else 0:
        small_g = filtered_df["Gender"].value_counts()
        smallest = small_g.idxmin()
        if is_small_sample(small_g.min()):
            small_sample_note(small_g.min(), f"the '{smallest}' gender group")

    c1, c2 = st.columns(2)
    with c1:
        show_chart_or_message(
            charts.treatment_pie(filtered_df),
            key="treatment-overview",
        )
    with c2:
        show_chart_or_message(
            charts.category_bar_chart(filtered_df, "Gender", "Gender Distribution")
        )

    c3, c4 = st.columns(2)
    with c3:
        show_chart_or_message(charts.age_histogram(filtered_df))
    with c4:
        show_chart_or_message(
            charts.category_bar_chart(
                filtered_df, "work_interfere", "Work Interference Distribution",
                x_label="Work Interference", order=WORK_INTERFERE_ORDER,
            )
        )

# ====================== B. DEMOGRAPHICS ======================
with tabs[1]:
    st.subheader("Demographics")

    show_chart_or_message(charts.age_histogram(filtered_df))

    c1, c2 = st.columns(2)
    with c1:
        show_chart_or_message(
            charts.category_bar_chart(filtered_df, "Gender", "Gender Distribution")
        )
    with c2:
        show_chart_or_message(
            charts.category_bar_chart(
                filtered_df, "Age Group", "Age Group Distribution",
                x_label="Age Group", order=AGE_GROUP_ORDER,
            )
        )

    top_c = top_countries(filtered_df, n=10)
    if top_c.empty:
        st.info("No data available for this chart with the current filters.")
    else:
        fig = charts.category_bar_chart  # not used directly; build custom horizontal chart below
        import plotly.express as px
        fig = px.bar(
            top_c.sort_values("Count"), x="Count", y="Country", orientation="h",
            title="Top 10 Countries by Respondent Count (after filtering)",
            text="Count", color_discrete_sequence=[charts.ACCENT],
        )
        fig.update_traces(textposition="outside", hovertemplate="%{y}<br>Count: %{x}<extra></extra>")
        fig.update_layout(
            template=charts.TEMPLATE,
            xaxis_title="Number of Respondents",
            yaxis_title="Country",
            title=dict(x=0.5, xanchor="center"),
        )
        fig.update_traces(cliponaxis=False)
        charts.apply_chart_layout(fig, height=480)
        _chart_counter["n"] += 1
        st.plotly_chart(
            fig,
            width="stretch",
            config={"responsive": True},
            key=f"chart_{_chart_counter['n']}",
        )

    show_chart_or_message(
        charts.category_bar_chart(
            filtered_df, "no_employees", "Company Size Distribution",
            x_label="Company Size (Number of Employees)",
        )
    )

# ====================== C. MENTAL HEALTH & TREATMENT ======================
with tabs[2]:
    st.subheader("Mental Health & Treatment")

    c1, c2 = st.columns(2)
    with c1:
        show_chart_or_message(
            charts.treatment_pie(filtered_df),
            key="treatment-treatment",
        )
    with c2:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "Age Group"), "Age Group",
                "Treatment Rate by Age Group", order=AGE_GROUP_ORDER,
            )
        )

    show_chart_or_message(
        charts.treatment_rate_bar(
            treatment_rate_by(filtered_df, "work_interfere"), "work_interfere",
            "Treatment Rate by Work Interference", x_label="Work Interference",
            order=WORK_INTERFERE_ORDER,
        )
    )

    c3, c4 = st.columns(2)
    with c3:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "Gender"), "Gender", "Treatment Rate by Gender"
            )
        )
    with c4:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "family_history"), "family_history",
                "Treatment Rate by Family History", x_label="Family History",
            )
        )

    c5, c6 = st.columns(2)
    with c5:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "remote_work"), "remote_work",
                "Treatment Rate by Remote Work", x_label="Remote Work",
            )
        )
    with c6:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "tech_company"), "tech_company",
                "Treatment Rate by Tech Company", x_label="Tech Company",
            )
        )

# ====================== D. WORKPLACE FACTORS ======================
with tabs[3]:
    st.subheader("Workplace Factors")

    c1, c2 = st.columns(2)
    with c1:
        show_chart_or_message(
            charts.category_bar_chart(
                filtered_df, "work_interfere", "Work Interference Distribution",
                x_label="Work Interference", order=WORK_INTERFERE_ORDER,
            )
        )
    with c2:
        show_chart_or_message(
            charts.category_bar_chart(filtered_df, "remote_work", "Remote Work Distribution")
        )

    c3, c4 = st.columns(2)
    with c3:
        show_chart_or_message(
            charts.category_bar_chart(filtered_df, "tech_company", "Tech Company Distribution")
        )
    with c4:
        show_chart_or_message(
            charts.category_bar_chart(
                filtered_df, "no_employees", "Company Size Distribution",
                x_label="Company Size (Number of Employees)",
            )
        )

    c5, c6 = st.columns(2)
    with c5:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "tech_company"), "tech_company",
                "Treatment Rate by Tech Company", x_label="Tech Company",
            )
        )
    with c6:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "remote_work"), "remote_work",
                "Treatment Rate by Remote Work", x_label="Remote Work",
            )
        )

    show_chart_or_message(
        charts.treatment_rate_bar(
            treatment_rate_by(filtered_df, "work_interfere"), "work_interfere",
            "Treatment Rate by Work Interference", x_label="Work Interference",
            order=WORK_INTERFERE_ORDER,
        )
    )

# ====================== E. SUPPORT & RESOURCES ======================
with tabs[4]:
    st.subheader("Support & Resources")

    c1, c2 = st.columns(2)
    with c1:
        show_chart_or_message(
            charts.category_bar_chart(filtered_df, "benefits", "Mental Health Benefits Distribution")
        )
    with c2:
        show_chart_or_message(
            charts.category_bar_chart(filtered_df, "care_options", "Care Options Distribution")
        )

    c3, c4 = st.columns(2)
    with c3:
        show_chart_or_message(
            charts.category_bar_chart(filtered_df, "wellness_program", "Wellness Program Distribution")
        )
    with c4:
        show_chart_or_message(
            charts.category_bar_chart(filtered_df, "seek_help", "Help-Seeking Resources Distribution",
                                       x_label="Seek Help")
        )

    c5, c6 = st.columns(2)
    with c5:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "benefits"), "benefits",
                "Treatment Rate by Mental Health Benefits", x_label="Benefits",
            )
        )
    with c6:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "care_options"), "care_options",
                "Treatment Rate by Care Options", x_label="Care Options",
            )
        )

    c7, c8 = st.columns(2)
    with c7:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "wellness_program"), "wellness_program",
                "Treatment Rate by Wellness Program", x_label="Wellness Program",
            )
        )
    with c8:
        show_chart_or_message(
            charts.treatment_rate_bar(
                treatment_rate_by(filtered_df, "seek_help"), "seek_help",
                "Treatment Rate by Help-Seeking Resources", x_label="Seek Help",
            )
        )

# ====================== F. MULTIVARIATE ANALYSIS ======================
with tabs[5]:
    st.subheader("Detailed Multivariate Analysis")
    st.caption(
        "These charts combine two factors at once. Small combinations may have very "
        "few respondents -- always check the sample size before drawing conclusions."
    )

    show_chart_or_message(
        charts.grouped_treatment_rate_bar(
            treatment_rate_by_two(filtered_df, "Gender", "family_history"),
            "Gender", "family_history",
            "Treatment Rate by Gender and Family History",
        )
    )

    show_chart_or_message(
        charts.grouped_treatment_rate_bar(
            treatment_rate_by_two(filtered_df, "Age Group", "Gender"),
            "Age Group", "Gender",
            "Treatment Rate by Age Group and Gender",
        )
    )

    show_chart_or_message(
        charts.grouped_treatment_rate_bar(
            treatment_rate_by_two(filtered_df, "work_interfere", "Gender"),
            "work_interfere", "Gender",
            "Treatment Rate by Work Interference and Gender",
        )
    )

    show_chart_or_message(
        charts.grouped_treatment_rate_bar(
            treatment_rate_by_two(filtered_df, "benefits", "care_options"),
            "benefits", "care_options",
            "Treatment Rate by Mental Health Benefits and Care Options",
        )
    )

    show_chart_or_message(
        charts.grouped_treatment_rate_bar(
            treatment_rate_by_two(filtered_df, "work_interfere", "remote_work"),
            "work_interfere", "remote_work",
            "Treatment Rate by Work Interference and Remote Work",
        )
    )

    st.markdown("#### Correlation Analysis")
    corr = correlation_matrix(filtered_df)
    show_chart_or_message(
        charts.correlation_heatmap(corr),
        empty_message="Correlation heatmap not available for the current filter selection.",
    )
    st.caption(
        "Correlation indicates association between variables and does not imply causation. "
        "Only a small set of meaningful numeric / binary-encoded variables "
        "(Age, self_employed, family_history, treatment, remote_work, tech_company) is used here."
    )

# ====================== G. KEY FINDINGS ======================
with tabs[6]:
    st.subheader("Key Findings")

    st.markdown("#### Overall EDA Findings")
    st.markdown(
        """
- Family history showed the strongest association with treatment among the selected key variables.
- Treatment rates varied considerably across work-interference categories.
- Mental-health benefits and care options showed differences in treatment-reporting percentages.
- Treatment patterns varied across age and gender groups.

*These findings are based on the overall exploratory data analysis and are stated in neutral,
associative language. No causal relationship is implied or claimed.*
        """
    )

    st.markdown("#### Current Filtered-View Findings")
    cur_treatment_rate = kpis["treatment_rate"]
    cur_family_rate = kpis["family_history_rate"]
    cur_n = kpis["total_respondents"]

    findings = [
        f"For the current selection ({cur_n:,} respondents), the treatment rate is "
        f"{cur_treatment_rate:.2f}%." if pd.notna(cur_treatment_rate) else
        "Treatment rate could not be computed for the current selection.",
        f"{cur_family_rate:.2f}% of respondents in this view report a family history of "
        f"mental illness." if pd.notna(cur_family_rate) else
        "Family history rate could not be computed for the current selection.",
    ]
    for f in findings:
        st.markdown(f"- {f}")

    if is_small_sample(cur_n):
        st.markdown(
            f'<div class="caution-note">⚠️ The current filtered view has a small sample size '
            f'(n = {cur_n}). Treat the figures above as indicative only.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Business Insights")
    st.markdown(
        """
- **Strengthening mental-health support:** Consistent, clearly communicated benefits and care
  options are associated with different treatment-seeking percentages, suggesting a possible
  opportunity area.
- **Improving awareness and accessibility:** Employees with clearer access to wellness programs
  and help-seeking resources show different treatment patterns in this dataset.
- **Understanding work interference:** Respondents reporting higher work interference show
  different treatment rates, which may warrant closer attention to workload and support policies.
- **Considering family-history patterns:** Family history is associated with treatment-seeking
  behavior more strongly than most other variables examined here.
- **Examining demographic differences:** Treatment rates differ across age and gender groups,
  which may inform targeted, non-stigmatizing communication.
- **Improving workplace support:** Overall patterns suggest continued investment in accessible,
  low-friction mental-health resources may be valuable.

*These are exploratory, descriptive observations from survey data -- not causal claims,
individual predictions, or diagnoses.*
        """
    )

st.markdown("---")

# ----------------------------------------------------------------------
# DOWNLOAD FILTERED DATA
# ----------------------------------------------------------------------
st.subheader("Download Filtered Data")
csv_bytes = filtered_df.drop(columns=["Age Group"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv_bytes,
    file_name="filtered_mental_health_data.csv",
    mime="text/csv",
    use_container_width=False,
)

# ----------------------------------------------------------------------
# ABOUT THE DATASET
# ----------------------------------------------------------------------
with st.expander("📊 About the Dataset"):
    info = dataset_overview(raw_df)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Records", f"{info['n_records']:,}")
    d2.metric("Columns", info["n_columns"])
    d3.metric("Age Range", f"{info['age_min']}–{info['age_max']}")
    d4.metric("Countries", info["n_countries"])

    e1, e2, e3 = st.columns(3)
    e1.metric("Gender Categories", info["n_gender_categories"])
    e2.metric("Missing Values", info["missing_values"])
    e3.metric("Duplicate Records", info["duplicate_records"])

st.caption(
    "Employee Mental Health Analytics Dashboard · Exploratory Data Analysis project · "
    "No machine learning, prediction, or diagnosis is performed by this application."
)
