"""
data_loader.py
---------------
Responsible for:
- Loading the cleaned survey CSV
- Data type conversion (Timestamp -> datetime, Age -> numeric)
- Age Group creation
- Basic safety checks (missing file / missing columns)

The original dataset is never modified in place -- every function here
returns a new DataFrame.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# CONFIGURATION -- change the dataset path/filename here only.
# ---------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mental_health_cleaned.csv"

# Columns the app relies on. Used for a friendly error message if the
# uploaded / configured CSV does not match the expected schema.
REQUIRED_COLUMNS = [
    "Timestamp", "Age", "Country", "self_employed", "family_history",
    "treatment", "work_interfere", "no_employees", "remote_work",
    "tech_company", "benefits", "care_options", "wellness_program",
    "seek_help", "anonymity", "leave", "mental_health_consequence",
    "phys_health_consequence", "coworkers", "supervisor",
    "mental_health_interview", "phys_health_interview",
    "mental_vs_physical", "obs_consequence", "comments", "Gender", "state",
]

AGE_GROUP_ORDER = ["18-25", "26-35", "36-45", "46-55", "56+"]
WORK_INTERFERE_ORDER = ["Never", "Rarely", "Sometimes", "Often", "Unknown"]


def _make_age_group(age_series: pd.Series) -> pd.Series:
    """Bucket numeric Age into the fixed, logically-ordered age groups."""
    bins = [17, 25, 35, 45, 55, np.inf]
    labels = AGE_GROUP_ORDER
    groups = pd.cut(age_series, bins=bins, labels=labels, right=True)
    return pd.Categorical(groups, categories=AGE_GROUP_ORDER, ordered=True)


def _normalize_work_interfere(series: pd.Series) -> pd.Series:
    """Make sure work_interfere only contains the five expected labels."""
    cleaned = series.fillna("Unknown").replace(
        {"": "Unknown", "Don't know": "Unknown", "NaN": "Unknown"}
    )
    cleaned = cleaned.where(cleaned.isin(WORK_INTERFERE_ORDER), "Unknown")
    return pd.Categorical(cleaned, categories=WORK_INTERFERE_ORDER, ordered=True)


class DatasetError(Exception):
    """Raised when the dataset cannot be loaded or is missing required columns."""


@st.cache_data(show_spinner="Loading survey data...")
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and lightly preprocess the cleaned mental-health survey dataset.

    Returns a new, cached DataFrame. Raises DatasetError with a clear
    message if the file is missing or malformed, so the caller (app.py)
    can show a graceful Streamlit error instead of crashing.
    """
    path = Path(path)

    if not path.exists():
        raise DatasetError(
            f"Dataset file not found at '{path}'. "
            "Please make sure 'mental_health_cleaned.csv' exists in the data/ folder."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read the dataset file: {exc}") from exc

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError(
            "The dataset is missing required column(s): "
            + ", ".join(missing_cols)
        )

    df = df.copy()

    # ---- dtype fixes ----
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Guard against corrupt / out-of-range ages without silently dropping rows.
    df = df[df["Age"].between(18, 100, inclusive="both") | df["Age"].isna()]

    # ---- feature engineering (derived, non-destructive) ----
    df["Age Group"] = _make_age_group(df["Age"])
    df["work_interfere"] = _normalize_work_interfere(df["work_interfere"])

    # Normalize a few string columns so " Yes" / "yes" style inconsistencies
    # don't create phantom filter categories.
    for col in ["Gender", "treatment", "remote_work", "tech_company",
                "family_history", "self_employed"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.reset_index(drop=True)
    return df


def dataset_overview(df: pd.DataFrame) -> dict:
    """Small summary dict used by the 'About the Dataset' expander."""
    return {
        "n_records": len(df),
        "n_columns": df.shape[1],
        "age_min": int(df["Age"].min()) if df["Age"].notna().any() else None,
        "age_max": int(df["Age"].max()) if df["Age"].notna().any() else None,
        "n_countries": df["Country"].nunique(),
        "n_gender_categories": df["Gender"].nunique(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_records": int(df.duplicated().sum()),
    }
