"""
calculations.py
----------------
Responsible for:
- Top KPI calculations
- Treatment-rate / grouped-rate calculations
- Correlation matrix preparation
- Small-sample-size warning helper

Every function is written to be safe on an EMPTY dataframe -- callers in
app.py should still check `filtered_df.empty` before rendering, but these
helpers never raise on an empty input.

Do not interpret any correlation output produced here as causation.
"""

import numpy as np
import pandas as pd

SMALL_SAMPLE_THRESHOLD = 30


def rate_yes(df: pd.DataFrame, column: str, positive_value: str = "Yes") -> float:
    """Percentage (0-100) of rows where `column` == positive_value. Safe on empty df."""
    if df.empty or column not in df.columns:
        return np.nan
    return round((df[column] == positive_value).mean() * 100, 2)


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute the five top-level KPI cards from the (already filtered) dataframe."""
    if df.empty:
        return {
            "total_respondents": 0,
            "treatment_rate": np.nan,
            "average_age": np.nan,
            "remote_work_rate": np.nan,
            "family_history_rate": np.nan,
        }
    return {
        "total_respondents": int(len(df)),
        "treatment_rate": rate_yes(df, "treatment"),
        "average_age": round(df["Age"].mean(), 2) if df["Age"].notna().any() else np.nan,
        "remote_work_rate": rate_yes(df, "remote_work"),
        "family_history_rate": rate_yes(df, "family_history"),
    }


def value_counts_with_pct(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a tidy DataFrame with count and percentage for each category."""
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "Count", "Percentage"])

    counts = df[column].value_counts(dropna=False)
    total = counts.sum()
    out = counts.rename_axis(column).reset_index(name="Count")
    out["Percentage"] = (out["Count"] / total * 100).round(2)
    return out


def treatment_rate_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Treatment RATE (%) grouped by a single categorical column.
    Returns columns: [group_col, 'Treatment Rate (%)', 'Respondents'].
    Safe on empty df / group with zero rows.
    """
    cols = [group_col, "Treatment Rate (%)", "Respondents"]
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=cols)

    grouped = (
        df.groupby(group_col, dropna=False, observed=True)["treatment"]
        .apply(lambda s: (s == "Yes").mean() * 100 if len(s) else np.nan)
        .round(2)
        .reset_index(name="Treatment Rate (%)")
    )
    counts = df.groupby(group_col, dropna=False, observed=True).size().reset_index(name="Respondents")
    result = grouped.merge(counts, on=group_col, how="left")
    return result[cols]


def treatment_rate_by_two(df: pd.DataFrame, group_col_1: str, group_col_2: str) -> pd.DataFrame:
    """
    Treatment RATE (%) grouped by TWO categorical columns -- used for the
    multivariate analysis tab (grouped bar charts).
    """
    cols = [group_col_1, group_col_2, "Treatment Rate (%)", "Respondents"]
    if df.empty or group_col_1 not in df.columns or group_col_2 not in df.columns:
        return pd.DataFrame(columns=cols)

    grouped = (
        df.groupby([group_col_1, group_col_2], dropna=False, observed=True)["treatment"]
        .apply(lambda s: (s == "Yes").mean() * 100 if len(s) else np.nan)
        .round(2)
        .reset_index(name="Treatment Rate (%)")
    )
    counts = (
        df.groupby([group_col_1, group_col_2], dropna=False, observed=True)
        .size()
        .reset_index(name="Respondents")
    )
    result = grouped.merge(counts, on=[group_col_1, group_col_2], how="left")
    return result[cols]


def top_countries(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N countries by respondent count, computed AFTER filtering."""
    if df.empty:
        return pd.DataFrame(columns=["Country", "Count"])
    out = (
        df["Country"].value_counts().head(n).rename_axis("Country").reset_index(name="Count")
    )
    return out


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix on a small, deliberately chosen set of meaningful
    numeric / binary-encoded variables (NOT every categorical column).
    Correlation indicates association only -- never causation.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    binary_map = {"Yes": 1, "No": 0}
    for col in ["self_employed", "family_history", "treatment", "remote_work", "tech_company"]:
        if col in work.columns:
            work[col] = work[col].map(binary_map)

    subset_cols = [c for c in
                   ["Age", "self_employed", "family_history", "treatment",
                    "remote_work", "tech_company"] if c in work.columns]

    numeric = work[subset_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.dropna(how="all").empty:
        return pd.DataFrame()

    return numeric.corr().round(2)


def is_small_sample(n: int, threshold: int = SMALL_SAMPLE_THRESHOLD) -> bool:
    """True when a group's sample size is small enough to warrant a caution note."""
    return 0 < n < threshold
