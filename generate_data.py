"""
Synthetic sample-data generator for the Employee Mental Health Dashboard.

NOTE: No real cleaned dataset was provided with the request. This script
generates a synthetic dataset that matches the schema, dimensions, and
summary statistics described in the project brief (1259 rows, 27 columns,
Age 18-72, treatment rate ~50.6%, remote work rate ~29.86%,
family history rate ~39.08%, average age ~32.07) so the dashboard is fully
runnable end-to-end. Replace data/mental_health_cleaned.csv with the real
cleaned dataset for actual use -- the app does not depend on this script.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
N = 1259

countries_pool = (
    ["United States"] * 60
    + ["United Kingdom"] * 12
    + ["Canada"] * 8
    + ["Germany"] * 5
    + ["Ireland"] * 3
    + ["Australia"] * 3
    + ["Netherlands"] * 3
    + ["France"] * 2
    + ["India"] * 2
    + ["Switzerland"] * 2
    + ["Brazil"] * 1
    + ["Sweden"] * 1
    + ["New Zealand"] * 1
    + ["Poland"] * 1
    + ["Belgium"] * 1
    + ["Italy"] * 1
)
countries = rng.choice(countries_pool, size=N)

us_states = ["CA", "WA", "NY", "TX", "IL", "OR", "MA", "PA", "OH", "MI",
             "NC", "GA", "VA", "CO", "MN", "FL", "AZ", "TN", "WI", "MD"]

genders = rng.choice(["Male", "Female", "Other"], size=N, p=[0.62, 0.35, 0.03])

age = np.clip(rng.normal(loc=31.5, scale=7.0, size=N), 18, 72).round().astype(int)
# Add a small tail of older respondents so the 56+ age group is represented,
# consistent with the stated Age range of 18-72 in the project brief.
tail_idx = rng.choice(N, size=max(1, int(N * 0.02)), replace=False)
age[tail_idx] = rng.integers(56, 73, size=len(tail_idx))

self_employed = rng.choice(["Yes", "No"], size=N, p=[0.12, 0.88])
family_history = rng.choice(["Yes", "No"], size=N, p=[0.3908, 0.6092])
remote_work = rng.choice(["Yes", "No"], size=N, p=[0.2986, 0.7014])
tech_company = rng.choice(["Yes", "No"], size=N, p=[0.80, 0.20])

work_interfere_opts = ["Never", "Rarely", "Sometimes", "Often", "Unknown"]

no_employees_opts = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
no_employees = rng.choice(no_employees_opts, size=N, p=[0.12, 0.18, 0.22, 0.18, 0.10, 0.20])

benefits_opts = ["Yes", "No", "Don't know"]
care_options_opts = ["Yes", "No", "Not sure"]
wellness_program_opts = ["Yes", "No", "Don't know"]
seek_help_opts = ["Yes", "No", "Don't know"]
anonymity_opts = ["Yes", "No", "Don't know"]
leave_opts = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]
consequence_opts = ["Yes", "No", "Maybe"]
coworkers_opts = ["Yes", "No", "Some of them"]
supervisor_opts = ["Yes", "No", "Some of them"]
interview_opts = ["Yes", "No", "Maybe"]
mental_vs_physical_opts = ["Yes", "No", "Don't know"]
obs_consequence_opts = ["Yes", "No"]

benefits = rng.choice(benefits_opts, size=N, p=[0.45, 0.30, 0.25])
care_options = rng.choice(care_options_opts, size=N, p=[0.40, 0.35, 0.25])
wellness_program = rng.choice(wellness_program_opts, size=N, p=[0.25, 0.55, 0.20])
seek_help = rng.choice(seek_help_opts, size=N, p=[0.30, 0.50, 0.20])
anonymity = rng.choice(anonymity_opts, size=N, p=[0.35, 0.15, 0.50])
leave = rng.choice(leave_opts, size=N, p=[0.15, 0.30, 0.25, 0.20, 0.10])
mental_health_consequence = rng.choice(consequence_opts, size=N, p=[0.25, 0.50, 0.25])
phys_health_consequence = rng.choice(consequence_opts, size=N, p=[0.10, 0.75, 0.15])
coworkers = rng.choice(coworkers_opts, size=N, p=[0.25, 0.20, 0.55])
supervisor = rng.choice(supervisor_opts, size=N, p=[0.40, 0.30, 0.30])
mental_health_interview = rng.choice(interview_opts, size=N, p=[0.05, 0.75, 0.20])
phys_health_interview = rng.choice(interview_opts, size=N, p=[0.15, 0.55, 0.30])
mental_vs_physical = rng.choice(mental_vs_physical_opts, size=N, p=[0.30, 0.30, 0.40])
obs_consequence = rng.choice(obs_consequence_opts, size=N, p=[0.15, 0.85])

# ---- Build treatment & work_interfere with realistic associations ----
work_interfere = np.empty(N, dtype=object)
treatment = np.empty(N, dtype=object)

for i in range(N):
    base = 0.0
    if family_history[i] == "Yes":
        base += 0.28
    base += rng.normal(0, 0.12)

    if base > 0.15:
        wi_p = [0.05, 0.10, 0.25, 0.40, 0.20]   # more likely Often/Sometimes
    elif base > -0.05:
        wi_p = [0.15, 0.25, 0.30, 0.15, 0.15]
    else:
        wi_p = [0.35, 0.30, 0.15, 0.05, 0.15]
    work_interfere[i] = rng.choice(work_interfere_opts, p=wi_p)

    treat_prob = 0.506
    if family_history[i] == "Yes":
        treat_prob += 0.22
    else:
        treat_prob -= 0.12
    if work_interfere[i] in ("Often", "Sometimes"):
        treat_prob += 0.10
    elif work_interfere[i] == "Never":
        treat_prob -= 0.10
    treat_prob = min(max(treat_prob, 0.03), 0.97)
    treatment[i] = "Yes" if rng.random() < treat_prob else "No"

# Timestamps spread across a survey window
start = datetime(2014, 8, 27)
timestamps = [start + timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 30, size=N)]

state = [rng.choice(us_states) if c == "United States" else "" for c in countries]
comments = [""] * N  # cleaned dataset: free-text comments mostly blank

df = pd.DataFrame({
    "Timestamp": timestamps,
    "Age": age,
    "Country": countries,
    "self_employed": self_employed,
    "family_history": family_history,
    "treatment": treatment,
    "work_interfere": work_interfere,
    "no_employees": no_employees,
    "remote_work": remote_work,
    "tech_company": tech_company,
    "benefits": benefits,
    "care_options": care_options,
    "wellness_program": wellness_program,
    "seek_help": seek_help,
    "anonymity": anonymity,
    "leave": leave,
    "mental_health_consequence": mental_health_consequence,
    "phys_health_consequence": phys_health_consequence,
    "coworkers": coworkers,
    "supervisor": supervisor,
    "mental_health_interview": mental_health_interview,
    "phys_health_interview": phys_health_interview,
    "mental_vs_physical": mental_vs_physical,
    "obs_consequence": obs_consequence,
    "comments": comments,
    "Gender": genders,
    "state": state,
})

df.to_csv("/home/claude/employee-mental-health-dashboard/data/mental_health_cleaned.csv", index=False)

print("Rows:", len(df), "Cols:", df.shape[1])
print("Treatment rate:", (df.treatment == "Yes").mean())
print("Remote work rate:", (df.remote_work == "Yes").mean())
print("Family history rate:", (df.family_history == "Yes").mean())
print("Average age:", df.Age.mean())
print("Gender categories:", df.Gender.unique())
print("Duplicates:", df.duplicated().sum())
print("Missing values:", df.isna().sum().sum())
