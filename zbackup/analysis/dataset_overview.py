
# Import libraries for dataset overview
import pandas as pd
import csv

from pathlib import Path

# Define the original dataset folder
DATASET_FOLDER = Path("dataset/MachineLearningCVE")

# Define the output folder
OUTPUT_FOLDER = Path("dataset/eda_reports")

# Create output folder
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Find all original CSV files
csv_files = sorted(DATASET_FOLDER.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATASET_FOLDER}"
    )

print("=" * 65)
print("CICIDS2017 DATASET OVERVIEW")
print("=" * 65)

print(f"Number of CSV files: {len(csv_files)}")

dataset_summary = []
reference_columns = None

# Analyse each original CSV file
for csv_file in csv_files:

    print("\n" + "-" * 65)
    print(f"File: {csv_file.name}")
    print("-" * 65)

    # Read column names from the CSV header
    with open(
        csv_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        columns = next(csv.reader(file))

    # Remove unnecessary spaces from column names
    columns = [column.strip() for column in columns]

    # Count records without loading the complete dataset
    with open(
        csv_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        total_rows = sum(1 for _ in csv.reader(file)) - 1

    # Calculate file size in MB
    file_size_mb = csv_file.stat().st_size / (1024 ** 2)

    # Check whether columns match the first CSV
    if reference_columns is None:
        reference_columns = columns
        column_status = "Reference file"
    else:
        column_status = (
            "Consistent"
            if columns == reference_columns
            else "Different"
        )

    # Store dataset information
    dataset_summary.append({
        "File": csv_file.name,
        "Total Records": total_rows,
        "Total Columns": len(columns),
        "File Size (MB)": round(file_size_mb, 2),
        "Column Status": column_status
    })

    print(f"Total records: {total_rows:,}")
    print(f"Total columns: {len(columns)}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Column status: {column_status}")

# Create summary dataframe
summary_df = pd.DataFrame(dataset_summary)

# Calculate total dataset records
total_records = summary_df["Total Records"].sum()

# Display final summary
print("\n" + "=" * 65)
print("OVERALL DATASET SUMMARY")
print("=" * 65)

print(summary_df.to_string(index=False))

print(f"\nTotal CSV files: {len(csv_files)}")
print(f"Total records: {total_records:,}")

# Save dataset overview report
output_file = OUTPUT_FOLDER / "file_overview.csv"

summary_df.to_csv(
    output_file,
    index=False
)

print(f"\nReport saved to: {output_file}")
print("Dataset overview completed.")
