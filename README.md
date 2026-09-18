# Employee Mental Health Analytics Dashboard

## Overview

An interactive Streamlit dashboard for exploring an employee mental-health
survey dataset. The dashboard lets users dynamically filter respondents by
demographic and workplace attributes and see KPIs and Plotly charts update
in real time. This is an **exploratory data analysis (EDA) project** —
it contains **no machine learning model, classifier, regression model, or
prediction pipeline** of any kind.

> **Note on the sample data:** No cleaned dataset file was supplied with this
> project, so `data/mental_health_cleaned.csv` currently contains a
> **synthetically generated sample dataset** built to match the schema and
> summary statistics described in the project brief (1,259 rows, 27 columns,
> Age 18–72, ~50–57% treatment rate, ~30% remote-work rate, ~40% family-history
> rate). **Replace this file with your real cleaned dataset** (same column
> names) before using the dashboard for actual analysis — no code changes are
> required as long as the column names match.

## Business Objective

To analyze employee mental-health survey data and identify demographic and
workplace factors associated with seeking mental-health treatment, so that
organizations can better understand gaps in mental-health support and make
data-informed improvements to workplace well-being initiatives.

## Dataset

- **Rows:** 1,259
- **Columns:** 27
- **Duplicate rows:** 0
- **Missing values:** 0
- **Age range:** 18–72
- **Gender categories:** Female, Male, Other

Key columns include `Age`, `Gender`, `Country`, `family_history`, `treatment`,
`work_interfere`, `no_employees`, `remote_work`, `tech_company`, `benefits`,
`care_options`, `wellness_program`, `seek_help`, `anonymity`, `leave`, and
several workplace-consequence / interview-related fields.

## EDA

The underlying exploratory analysis (see your source notebook for the full
workup) covers:

- **Univariate analysis** — distributions of Age, Gender, Country, company
  size, work interference, and treatment.
- **Bivariate analysis** — treatment rate by gender, family history, age
  group, work interference, remote work, and tech company.
- **Multivariate analysis** — treatment rate broken down by two factors at
  once (e.g. gender × family history, work interference × remote work).
- **Correlation heatmap** — association between Age and a small set of
  binary-encoded workplace/demographic variables (correlation, not
  causation).
- **Pair plot** — considered during EDA but intentionally **omitted** from
  the interactive dashboard in favor of clearer, filter-aware Plotly charts;
  it remains part of the notebook-based EDA.

## Dashboard Features

- Sidebar filters for Gender, Age Group, Country, Treatment, Remote Work,
  Tech Company, Family History, and Work Interference — every filter
  includes an "All" option and a one-click **Reset Filters** button.
- All filters combine into a single centralized `filtered_df` used
  consistently across every KPI and chart.
- Five dynamic KPI cards: Total Respondents, Treatment Rate, Average Age,
  Remote Work Rate, Family History Rate.
- Seven sections/tabs: Overview, Demographics, Mental Health & Treatment,
  Workplace Factors, Support & Resources, Multivariate Analysis, Key
  Findings.
- Fully interactive Plotly charts (hover, zoom, legends) with counts and
  percentages shown as labels or hover text.
- Graceful empty-state handling — filter combinations that return zero rows
  show a friendly message instead of crashing.
- Small-sample-size cautionary notes (n < 30) shown where relevant.
- Current filter summary panel.
- One-click download of the currently filtered data as CSV.
- "About the Dataset" and "About / Business Objective" expanders.
- Key Findings tab separating **overall EDA findings** from **current
  filtered-view findings**, written in neutral, non-causal language.

## Tech Stack

- Python 3.10+
- Pandas
- NumPy
- Streamlit
- Plotly

## Project Structure

```
employee-mental-health-dashboard/
│
├── app.py
│
├── data/
│   └── mental_health_cleaned.csv
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── filters.py
│   ├── calculations.py
│   └── charts.py
│
├── assets/                   # optional (e.g. logo.png)
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── README.md
├── .gitignore
└── generate_data.py          # optional: regenerates the sample dataset
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deployment (Streamlit Community Cloud)

1. Push this project (including `data/mental_health_cleaned.csv`) to a public
   or private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select the repository, branch, and set the main file
   path to `app.py`.
4. Click **"Deploy"**. No secrets or API keys are required for this
   application.
5. Once deployed, Streamlit Community Cloud will automatically re-deploy on
   every push to the connected branch.

The app uses only relative paths (`Path(__file__).parent / "data" / ...`), so
it runs identically on any machine or cloud environment without modification.

## Important Analytical Notes

- This dashboard performs **descriptive, exploratory analytics only**.
- It does **not** train, evaluate, or apply any machine learning model.
- It does **not** diagnose mental-health conditions or rank individual
  employees.
- Correlation and association results are never described as causal.
