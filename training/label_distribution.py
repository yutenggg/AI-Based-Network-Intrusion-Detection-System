# Import libraries for label distribution analysis
import pandas as pd

from pathlib import Path
from collections import Counter

# Define the original dataset folder
DATASET_FOLDER = Path("dataset/MachineLearningCVE")

# Define the output folder
OUTPUT_FOLDER = Path("dataset/eda_reports")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Find all original CSV files
csv_files = sorted(DATASET_FOLDER.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATASET_FOLDER}"
    )

# Store label counts
overall_label_counts = Counter()
file_label_results = []

print("=" * 65)
print("CICIDS2017 LABEL DISTRIBUTION ANALYSIS")
print("=" * 65)

# Analyse labels in each original CSV file
for csv_file in csv_files:

    print(f"\nAnalysing: {csv_file.name}")

    file_label_counts = Counter()

    # Read large CSV files in smaller chunks
    for chunk in pd.read_csv(
        csv_file,
        chunksize=100000,
        encoding="utf-8-sig"
    ):

        # Remove unnecessary spaces from column names
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(
                f"Label column not found in {csv_file.name}"
            )

        # Clean label names
        labels = chunk["Label"].astype(str).str.strip()

        # Count records for each label
        file_label_counts.update(labels.value_counts().to_dict())

    # Add file results to overall results
    overall_label_counts.update(file_label_counts)

    for label, count in file_label_counts.items():

        file_label_results.append({
            "File": csv_file.name,
            "Attack Type": label,
            "Records": count
        })

    print(f"Number of labels: {len(file_label_counts)}")
    print(f"Total records: {sum(file_label_counts.values()):,}")

# Create overall label distribution report
overall_df = pd.DataFrame(
    overall_label_counts.items(),
    columns=["Attack Type", "Records"]
)

# Calculate percentage for each label
total_records = overall_df["Records"].sum()

overall_df["Percentage"] = (
    overall_df["Records"] / total_records * 100
).round(2)

# Sort labels by record count
overall_df = overall_df.sort_values(
    by="Records",
    ascending=False
)

# Create per-file label distribution report
file_df = pd.DataFrame(file_label_results)

# Save reports
overall_output = OUTPUT_FOLDER / "label_distribution_overall.csv"
file_output = OUTPUT_FOLDER / "label_distribution_by_file.csv"

overall_df.to_csv(overall_output, index=False)
file_df.to_csv(file_output, index=False)

# Display overall results
print("\n" + "=" * 65)
print("OVERALL LABEL DISTRIBUTION")
print("=" * 65)

print(overall_df.to_string(index=False))

print(f"\nTotal records: {total_records:,}")
print(f"Total attack types: {len(overall_df)}")

print(f"\nOverall report saved to: {overall_output}")
print(f"Per-file report saved to: {file_output}")

print("\nLabel distribution analysis completed.")
