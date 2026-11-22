
# Verify the cleaned CICIDS2017 dataset

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


DATASET_PATH = Path("dataset/processed/cleaned_CICIDS2017.csv")
CHART_PATH = Path("dataset/eda_reports/charts/cleaning_verification.png")
TEMP_DB_PATH = Path("dataset/processed/cleaning_verification_temp.sqlite")

CHUNK_SIZE = 100000


def main():
    print("Starting Data Cleaning Verification...")

    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TEMP_DB_PATH.exists():
        TEMP_DB_PATH.unlink()

    connection = sqlite3.connect(TEMP_DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE row_hashes (
            row_hash TEXT PRIMARY KEY
        )
    """)

    total_records = 0
    missing_values = 0
    infinite_values = 0
    duplicate_records = 0
    feature_count = 0
    label_counts = {}

    try:
        for chunk_number, df in enumerate(
            pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE),
            start=1
        ):
            print(f"Verifying chunk {chunk_number}...")

            total_records += len(df)

            if "Label" not in df.columns:
                raise ValueError("The Label column is missing.")

            feature_count = len(df.columns) - 1

            missing_values += int(df.isna().sum().sum())

            numeric_columns = df.select_dtypes(
                include=[np.number]
            ).columns

            if len(numeric_columns) > 0:
                infinite_values += int(
                    np.isinf(df[numeric_columns]).sum().sum()
                )

            counts = df["Label"].value_counts(dropna=False)

            for label, count in counts.items():
                label_counts[str(label)] = (
                    label_counts.get(str(label), 0) + int(count)
                )

            row_hashes = pd.util.hash_pandas_object(
                df,
                index=False
            )

            hash_values = [
                (str(int(value)),)
                for value in row_hashes
            ]

            previous_changes = connection.total_changes

            cursor.executemany(
                "INSERT OR IGNORE INTO row_hashes (row_hash) VALUES (?)",
                hash_values
            )

            inserted_records = (
                connection.total_changes - previous_changes
            )

            duplicate_records += len(df) - inserted_records

            connection.commit()

    finally:
        connection.close()

        if TEMP_DB_PATH.exists():
            TEMP_DB_PATH.unlink()

    print("\nData Cleaning Verification Completed!")
    print(f"Total Records: {total_records:,}")
    print(f"Total Features: {feature_count}")
    print(f"Total Columns: {feature_count + 1}")
    print(f"Missing Values: {missing_values:,}")
    print(f"Infinite Values: {infinite_values:,}")
    print(f"Duplicate Records: {duplicate_records:,}")
    print(f"Number of Labels: {len(label_counts)}")

    print("\nLabel Distribution:")

    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count:,}")

    results = {
        "Missing Values": missing_values,
        "Infinite Values": infinite_values,
        "Duplicate Records": duplicate_records
    }

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        results.keys(),
        results.values(),
        color=["steelblue", "darkorange", "seagreen"],
        width=0.5
    )

    plt.title("Data Cleaning Verification Results")
    plt.ylabel("Number of Records")
    plt.xlabel("Verification Category")
    plt.xticks(rotation=10)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, results.values()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=300)
    plt.show()
    plt.close()

    print(f"\nChart saved successfully: {CHART_PATH}")


if __name__ == "__main__":
    main()
