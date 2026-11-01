"""Step 1 - Data registration.

Reads tourism.csv from the repository data folder, checks that every expected
column is present and prints a short summary. The job fails (exit code 1) if
the file is missing, empty or does not have the expected schema, so a bad
dataset never reaches the next stage of the pipeline.
"""
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        sys.exit("ERROR: dataset is empty")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing expected columns: {missing}")

    print(f"Dataset registered from: {DATA_PATH}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("All expected columns are present.\n")
    print("Column types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column (only columns with gaps):")
    gaps = df.isnull().sum()
    print(gaps[gaps > 0].to_string() if (gaps > 0).any() else "none")
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
