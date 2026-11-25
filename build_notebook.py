import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md("""# NYC Restaurant Health Inspections — Data Cleaning Capstone

**Skills applied:** handling missing values, scaling/parsing inconsistent text, parsing dates,
fixing data types, removing duplicates, detecting and handling outliers.

**Scenario:** A city health department's restaurant inspection export is messy — inconsistent
borough/cuisine spelling and casing, four different date formats, impossible inspection scores,
duplicate rows, and missing values. This notebook turns the raw export into an analysis-ready
dataset and documents every decision.

**Dataset:** Synthetically generated to mimic the structure and messiness of the real
[NYC Restaurant Inspection Results](https://www.kaggle.com/datasets/new-york-city/nyc-restaurant-inspection-results)
Kaggle dataset, so the notebook can run fully offline and reproducibly.
"""))

cells.append(md("## 1. Load the raw data and get a first look"))
cells.append(code("""import pandas as pd
import numpy as np

raw = pd.read_csv("data/raw_restaurant_inspections.csv")
print(raw.shape)
raw.head(10)
"""))

cells.append(code("""raw.info()
"""))

cells.append(md("""## 2. Scope the damage

Before fixing anything, quantify every problem: missing values, duplicates, and the
unique (messy) values in each categorical column."""))

cells.append(code("""missing = raw.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values per column:")
print(missing)

print("\\nExact duplicate rows:", raw.duplicated().sum())

print("\\nUnique raw 'borough' values:", sorted(raw['borough'].astype(str).unique()))
print("\\nUnique raw 'cuisine_type' values:", sorted(raw['cuisine_type'].astype(str).unique()))
"""))

cells.append(md("""## 3. Remove exact duplicates

Duplicate inspection rows would double-count restaurants in any aggregation, so they're
dropped first, before any other transformation touches the data."""))

cells.append(code("""df = raw.drop_duplicates().copy()
print(f"Dropped {len(raw) - len(df)} duplicate rows -> {len(df)} rows remain")
"""))

cells.append(md("""## 4. Standardize text columns

`borough` and `cuisine_type` have the same real values typed in different cases, with
stray whitespace. We normalize by stripping whitespace, upper-casing for comparison,
then mapping back to a clean, title-cased canonical label."""))

cells.append(code("""def clean_category(series: pd.Series, canonical: list[str]) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    lookup = {c.upper(): c for c in canonical}
    # also catch common abbreviations seen in this dataset
    manual_fixes = {"BKLYN": "Brooklyn", "S.I.": "Staten Island", "AMER.": "American", "JAP.": "Japanese"}
    def map_value(v):
        key = v.upper()
        if key in manual_fixes:
            return manual_fixes[key]
        return lookup.get(key, v)
    return cleaned.apply(map_value)

boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
cuisines = ["Italian", "Chinese", "American", "Mexican", "Indian", "Japanese", "Pizza"]

df["borough"] = clean_category(df["borough"], boroughs)
df["cuisine_type"] = clean_category(df["cuisine_type"], cuisines)

print("Boroughs after cleaning:", sorted(df["borough"].unique()))
print("Cuisines after cleaning:", sorted(df["cuisine_type"].unique()))
"""))

cells.append(md("""## 5. Parse the four inconsistent date formats

`inspection_date` mixes `YYYY-MM-DD`, `MM/DD/YYYY`, `DD-MM-YYYY`, and `YYYY/MM/DD`.
`pd.to_datetime` with `format="mixed"` infers the format per-row, which is the safest
option when formats aren't labeled."""))

cells.append(code("""df["inspection_date"] = pd.to_datetime(df["inspection_date"], format="mixed", dayfirst=False, errors="coerce")
print("Unparseable dates:", df["inspection_date"].isna().sum())
df["inspection_date"].dt.year.value_counts().sort_index()
"""))

cells.append(md("""## 6. Fix the `violations_count` column (mixed types + missing values)

One stray row stores the word `"two"` instead of a number. We coerce to numeric
(turning unparseable values into `NaN`), then decide how to handle the missing values:
here we impute with the **median**, which is robust to the outlier scores we'll handle
next, and we keep a flag column so the imputation is transparent rather than silently
hiding missing data."""))

cells.append(code("""df["violations_count"] = pd.to_numeric(df["violations_count"], errors="coerce")

df["violations_count_was_missing"] = df["violations_count"].isna()
median_violations = df["violations_count"].median()
df["violations_count"] = df["violations_count"].fillna(median_violations)
df["violations_count"] = df["violations_count"].astype(int)

print(f"Imputed {df['violations_count_was_missing'].sum()} missing violation counts with median={median_violations}")
"""))

cells.append(md("""## 7. Handle impossible / outlier inspection scores

A valid inspection score is 0-100. We found values of `-5` and `999`, which are data
entry errors, not real extreme scores. Rather than silently imputing, we clip them to
`NaN` and then flag + impute with the column median, which keeps the row (and its other
valid fields) usable while not distorting the score distribution."""))

cells.append(code("""invalid_score_mask = (df["score"] < 0) | (df["score"] > 100)
print("Invalid scores found:", invalid_score_mask.sum())

df.loc[invalid_score_mask, "score"] = np.nan
df["score_was_invalid"] = invalid_score_mask
df["score"] = df["score"].fillna(df["score"].median()).astype(int)

df["score"].describe()
"""))

cells.append(md("""## 8. Handle missing `restaurant_name` and `grade`

- `restaurant_name` missing -> fill with a clear placeholder (`"Unknown"`) since we can't
  invent a name, and dropping the row would throw away a valid inspection record.
- `grade` missing -> many real inspections genuinely have no grade yet (pending), so we
  fill with the explicit category `"Not Yet Graded"` instead of guessing a letter grade."""))

cells.append(code("""df["restaurant_name"] = df["restaurant_name"].fillna("Unknown")
df["grade"] = df["grade"].fillna("Not Yet Graded")

df[["restaurant_name", "grade"]].isna().sum()
"""))

cells.append(md("## 9. Final validation — confirm the dataset is analysis-ready"))
cells.append(code("""assert df.duplicated().sum() == 0
assert df["score"].between(0, 100).all()
assert df["borough"].isin(boroughs).all()
assert df["cuisine_type"].isin(cuisines).all()
assert df["inspection_date"].notna().all()
assert df["violations_count"].dtype == int

print("All validation checks passed.")
df.info()
"""))

cells.append(md("## 10. Save the cleaned dataset"))
cells.append(code("""df.to_csv("data/cleaned_restaurant_inspections.csv", index=False)
print("Saved data/cleaned_restaurant_inspections.csv with", len(df), "rows")
df.head()
"""))

cells.append(md("""## 11. Quick before/after summary

A short, reusable summary of what changed, useful for a README or a PR description."""))

cells.append(code("""summary = pd.DataFrame({
    "metric": [
        "rows (raw)", "rows (cleaned)", "duplicate rows removed",
        "invalid scores fixed", "missing violations imputed",
        "missing names filled", "missing grades filled",
    ],
    "value": [
        len(raw), len(df), len(raw) - len(df.drop_duplicates()) if False else (raw.duplicated().sum()),
        int(invalid_score_mask.sum()), int(df['violations_count_was_missing'].sum()),
        int(raw['restaurant_name'].isna().sum()), int(raw['grade'].isna().sum()),
    ],
})
summary
"""))

n = nbf.v4.new_notebook()
n["cells"] = cells
n["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

with open("notebooks/data_cleaning_restaurant_inspections.ipynb", "w") as f:
    nbf.write(n, f)
print("Notebook written.")
