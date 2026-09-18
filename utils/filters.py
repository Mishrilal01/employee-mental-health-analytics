"""
filters.py
----------
Responsible for:
- Rendering the sidebar 'Dashboard Filters'
- Handling the 'Reset Filters' button via st.session_state
- Applying every selected filter together to produce ONE centralized
  filtered dataframe (filtered_df) that the rest of the app reuses.
"""

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from utils.data_loader import AGE_GROUP_ORDER, WORK_INTERFERE_ORDER

ALL = "All"

# Keys used in st.session_state for each widget -> also the reset target.
_DEFAULTS = {
    "flt_gender": ALL,
    "flt_age_group": ALL,
    "flt_country": [ALL],
    "flt_treatment": ALL,
    "flt_remote_work": ALL,
    "flt_tech_company": ALL,
    "flt_family_history": ALL,
    "flt_work_interfere": ALL,
}


@dataclass
class FilterSelections:
    gender: str = ALL
    age_group: str = ALL
    country: list = field(default_factory=lambda: [ALL])
    treatment: str = ALL
    remote_work: str = ALL
    tech_company: str = ALL
    family_history: str = ALL
    work_interfere: str = ALL


def _init_state():
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _reset_filters():
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default


def render_sidebar_filters(df: pd.DataFrame) -> FilterSelections:
    """Render every sidebar widget and return the current selections."""
    _init_state()

        st.sidebar.markdown(
            '<div class="sidebar-brand"><span class="sidebar-brand-mark">+</span>'
            '<div><strong>PEOPLE PULSE</strong><small>Wellbeing analytics</small></div></div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("### Refine the view")
    st.sidebar.caption("Use the controls below to explore patterns across the survey.")

    st.sidebar.button(
        "↺  Reset filters", use_container_width=True, on_click=_reset_filters
    )
    st.sidebar.markdown("---")

    gender_opts = [ALL] + sorted(df["Gender"].dropna().unique().tolist())
    st.sidebar.selectbox("Gender", gender_opts, key="flt_gender")

    age_group_opts = [ALL] + [g for g in AGE_GROUP_ORDER
                               if g in df["Age Group"].dropna().unique().tolist()]
    st.sidebar.selectbox("Age Group", age_group_opts, key="flt_age_group")

    country_opts = sorted(df["Country"].dropna().unique().tolist())
    st.sidebar.multiselect(
        "Country",
        options=[ALL] + country_opts,
        key="flt_country",
        help="Select 'All' or one/more specific countries.",
    )

    st.sidebar.selectbox("Treatment", [ALL, "Yes", "No"], key="flt_treatment")
    st.sidebar.selectbox("Remote Work", [ALL, "Yes", "No"], key="flt_remote_work")
    st.sidebar.selectbox("Tech Company", [ALL, "Yes", "No"], key="flt_tech_company")
    st.sidebar.selectbox("Family History", [ALL, "Yes", "No"], key="flt_family_history")

    wi_opts = [ALL] + [w for w in WORK_INTERFERE_ORDER
                        if w in df["work_interfere"].dropna().unique().tolist()]
    st.sidebar.selectbox("Work Interference", wi_opts, key="flt_work_interfere")

    return FilterSelections(
        gender=st.session_state["flt_gender"],
        age_group=st.session_state["flt_age_group"],
        country=st.session_state["flt_country"],
        treatment=st.session_state["flt_treatment"],
        remote_work=st.session_state["flt_remote_work"],
        tech_company=st.session_state["flt_tech_company"],
        family_history=st.session_state["flt_family_history"],
        work_interfere=st.session_state["flt_work_interfere"],
    )


def apply_filters(df: pd.DataFrame, sel: FilterSelections) -> pd.DataFrame:
    """Apply all selected filters together and return ONE filtered dataframe."""
    mask = pd.Series(True, index=df.index)

    if sel.gender != ALL:
        mask &= df["Gender"] == sel.gender

    if sel.age_group != ALL:
        mask &= df["Age Group"] == sel.age_group

    if sel.country and ALL not in sel.country:
        mask &= df["Country"].isin(sel.country)

    if sel.treatment != ALL:
        mask &= df["treatment"] == sel.treatment

    if sel.remote_work != ALL:
        mask &= df["remote_work"] == sel.remote_work

    if sel.tech_company != ALL:
        mask &= df["tech_company"] == sel.tech_company

    if sel.family_history != ALL:
        mask &= df["family_history"] == sel.family_history

    if sel.work_interfere != ALL:
        mask &= df["work_interfere"] == sel.work_interfere

    return df.loc[mask].copy()


def filter_summary(sel: FilterSelections) -> dict:
    """Human-readable summary of the current filter selection."""
    country_display = "All" if (not sel.country or ALL in sel.country) else ", ".join(sel.country)
    return {
        "Gender": sel.gender,
        "Age Group": sel.age_group,
        "Country": country_display,
        "Treatment": sel.treatment,
        "Remote Work": sel.remote_work,
        "Tech Company": sel.tech_company,
        "Family History": sel.family_history,
        "Work Interference": sel.work_interfere,
    }
