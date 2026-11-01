"""Step 2 - Data preparation.

Loads the dataset straight from the repository data folder, cleans it,
splits it into train / test sets and saves the four CSV files in the current
working directory (the repo root in the GitHub Actions job). The workflow then
hands these files to the training job as an artifact.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"
OUT_DIR = Path(".")          # workflow uploads Xtrain.csv etc. from the repo root
TARGET = "ProdTaken"
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)

    # 1. Remove columns that carry no predictive information
    df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

    # 2. Clean text columns: trim spaces and fix the inconsistent gender label
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # 3. Drop exact duplicates (identical customer profiles can leak across the split)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed: {before - len(df)}")
    print("Missing values left (imputed inside the model pipeline):",
          int(df.isnull().sum().sum()))

    # 4. Stratified train / test split (keeps the ~19% buyer rate in both sets)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    for name, obj in [("Xtrain", Xtrain), ("Xtest", Xtest), ("ytrain", ytrain), ("ytest", ytest)]:
        obj.to_csv(OUT_DIR / f"{name}.csv", index=False)

    print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")
    print("Buyer rate  train: %.3f | test: %.3f" % (ytrain.mean(), ytest.mean()))
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
