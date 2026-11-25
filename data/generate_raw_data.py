"""
Generates a deliberately messy 'Restaurant Health Inspections' raw dataset
to simulate the kind of real-world data a Data Cleaning project should tackle:
- inconsistent casing / whitespace
- mixed date formats
- missing values
- duplicate rows
- inconsistent category labels
- outlier / impossible values
- mixed data types in a numeric column
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n = 600

boroughs_clean = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
borough_variants = {
    "Manhattan": ["Manhattan", "manhattan", "MANHATTAN", " Manhattan", "Manhattan "],
    "Brooklyn": ["Brooklyn", "brooklyn", "BROOKLYN", "Bklyn", " Brooklyn"],
    "Queens": ["Queens", "queens", "QUEENS", "Queens "],
    "Bronx": ["Bronx", "bronx", "BRONX", " Bronx"],
    "Staten Island": ["Staten Island", "staten island", "STATEN ISLAND", "S.I."],
}

cuisines_clean = ["Italian", "Chinese", "American", "Mexican", "Indian", "Japanese", "Pizza"]
cuisine_variants = {
    "Italian": ["Italian", "italian", "ITALIAN", "Italian "],
    "Chinese": ["Chinese", "chinese", "CHINESE"],
    "American": ["American", "american", "AMERICAN", "Amer."],
    "Mexican": ["Mexican", "mexican", "MEXICAN"],
    "Indian": ["Indian", "indian", "INDIAN"],
    "Japanese": ["Japanese", "japanese", "JAPANESE", "Jap."],
    "Pizza": ["Pizza", "pizza", "PIZZA"],
}

grades_clean = ["A", "B", "C"]

rows = []
for i in range(n):
    rid = 1000 + i
    name = f"Restaurant {rng.integers(1, 350)}"
    borough_key = rng.choice(boroughs_clean)
    borough = rng.choice(borough_variants[borough_key])
    cuisine_key = rng.choice(cuisines_clean)
    cuisine = rng.choice(cuisine_variants[cuisine_key])

    # score: mostly 0-100, but inject some impossible negative / huge outliers
    base_score = rng.integers(0, 60)
    if rng.random() < 0.02:
        score = -5  # impossible negative score
    elif rng.random() < 0.02:
        score = 999  # impossible outlier
    else:
        score = base_score

    grade = rng.choice(grades_clean + [None], p=[0.5, 0.3, 0.15, 0.05])

    # inspection date in multiple formats
    year = rng.integers(2021, 2025)
    month = rng.integers(1, 13)
    day = rng.integers(1, 28)
    date_formats = [
        f"{year}-{month:02d}-{day:02d}",
        f"{month:02d}/{day:02d}/{year}",
        f"{day:02d}-{month:02d}-{year}",
        f"{year}/{month:02d}/{day:02d}",
    ]
    inspection_date = rng.choice(date_formats)

    # violations count, with some missing values and a stray string
    if rng.random() < 0.08:
        violations = np.nan
    elif rng.random() < 0.02:
        violations = "two"  # bad type
    else:
        violations = int(rng.integers(0, 12))

    rows.append(
        {
            "inspection_id": rid,
            "restaurant_name": name if rng.random() > 0.01 else None,
            "borough": borough,
            "cuisine_type": cuisine,
            "inspection_date": inspection_date,
            "score": score,
            "grade": grade,
            "violations_count": violations,
        }
    )

df = pd.DataFrame(rows)

# Inject ~25 exact duplicate rows
dupe_idx = rng.choice(df.index, size=25, replace=False)
df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

df.to_csv("/home/claude/kaggle-portfolio/01-data-cleaning-restaurant-inspections/data/raw_restaurant_inspections.csv", index=False)
print("Generated", len(df), "raw rows")
print(df.head(10))
