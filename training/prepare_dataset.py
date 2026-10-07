import pandas as pd
import numpy as np
from pathlib import Path

DATASET_FOLDER = Path("dataset/MachineLearningCVE")

PROCESSED_FOLDER = Path("dataset/processed")

PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

csv_files = list(DATASET_FOLDER.glob("*.csv"))

print("Number of CSV files found:", len(csv_files))

processed_dataframes = []

for file in csv_files:

    print("\n====================================")
    print("Processing:", file.name)
    print("====================================")

    df = pd.read_csv(file)

    print("Original rows:", len(df))
    print("Original columns:", len(df.columns))

    df.columns = df.columns.str.strip()

    before_missing = len(df)

    df = df.dropna()

    removed_missing = before_missing - len(df)

    print("Missing rows removed:", removed_missing)

    numeric_columns = df.select_dtypes(include=["number"]).columns

    infinity_mask = np.isinf(df[numeric_columns]).any(axis=1)

    infinity_rows = infinity_mask.sum()

    df = df[~infinity_mask]

    print("Infinite rows removed:", infinity_rows)

    before_duplicates = len(df)

    df = df.drop_duplicates()

    removed_duplicates = before_duplicates - len(df)

    print("Duplicate rows removed:", removed_duplicates)

    print("Cleaned rows:", len(df))


    processed_dataframes.append(df)


print("\n====================================")
print("Merging cleaned datasets...")
print("====================================")

data = pd.concat(
    processed_dataframes,
    ignore_index=True
)

print("\nChecking duplicates after merging...")

before_duplicates = len(data)

data = data.drop_duplicates()

removed_duplicates = before_duplicates - len(data)

print("Duplicates removed after merging:", removed_duplicates)

print("\n====================================")
print("Final cleaned dataset")
print("====================================")

print("Total rows:", len(data))
print("Total columns:", len(data.columns))

print("\nAttack labels:")

print(data["Label"].value_counts())

output_file = PROCESSED_FOLDER / "cleaned_CICIDS2017.csv"

print("\nSaving cleaned dataset...")

data.to_csv(output_file, index=False)

print("Saved to:", output_file)