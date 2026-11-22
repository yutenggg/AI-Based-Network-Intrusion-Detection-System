
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set dataset and output directories
dataset_dir = Path("dataset/MachineLearningCVE")
output_dir = Path("dataset/eda_reports/charts")
output_dir.mkdir(parents=True, exist_ok=True)

csv_files = sorted(dataset_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {dataset_dir}"
    )

overall_missing = {}
file_missing = {}
total_rows = 0

print("Starting Missing Values Analysis...")

# Analyze Missing Values in Original CSV Files

for csv_file in csv_files:
    print(f"Analyzing: {csv_file.name}")

    file_missing_count = 0
    file_rows = 0

    for chunk in pd.read_csv(
        csv_file,
        chunksize=100000,
        encoding="utf-8-sig"
    ):
        missing_counts = chunk.isna().sum()

        for column, count in missing_counts.items():
            overall_missing[column] = (
                overall_missing.get(column, 0) + count
            )

        file_missing_count += missing_counts.sum()
        file_rows += len(chunk)

    file_missing[csv_file.name] = file_missing_count
    total_rows += file_rows

print("\nMissing Values Analysis Completed!")

# Chart 1: Missing Values by Feature

missing_df = pd.DataFrame(
    list(overall_missing.items()),
    columns=["Feature", "Missing Values"]
)

nonzero_missing = missing_df[
    missing_df["Missing Values"] > 0
].sort_values("Missing Values", ascending=True)

if nonzero_missing.empty:
    fig1, ax1 = plt.subplots(figsize=(9, 3))

    ax1.text(
        0.5, 0.5,
        "No Missing Values Found",
        ha="center",
        va="center",
        fontsize=16
    )

    ax1.set_title("Missing Values by Feature in CICIDS2017")
    ax1.axis("off")

else:
    chart_height = max(2.8, len(nonzero_missing) * 0.7)

    fig1, ax1 = plt.subplots(
        figsize=(10, chart_height)
    )

    bars1 = ax1.barh(
        nonzero_missing["Feature"],
        nonzero_missing["Missing Values"]
    )

    ax1.set_xlabel("Number of Missing Values")
    ax1.set_ylabel("Feature")
    ax1.set_title("Missing Values by Feature in CICIDS2017")

    ax1.bar_label(
        bars1,
        labels=[
            f"{value:,}"
            for value in nonzero_missing["Missing Values"]
        ],
        padding=5
    )

    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

fig1.tight_layout()

fig1.savefig(
    output_dir / "missing_values_by_feature.png",
    dpi=300,
    bbox_inches="tight"
)

# Chart 2: Missing Values by Original CSV File

file_df = pd.DataFrame(
    list(file_missing.items()),
    columns=["File", "Missing Values"]
)

file_df = file_df.sort_values(
    "Missing Values",
    ascending=True
)

fig2, ax2 = plt.subplots(figsize=(12, 7))

bars2 = ax2.barh(
    file_df["File"],
    file_df["Missing Values"]
)

ax2.set_xlabel("Total Number of Missing Values")
ax2.set_ylabel("Original CSV File")
ax2.set_title("Missing Values by Original CICIDS2017 CSV File")

ax2.bar_label(
    bars2,
    labels=[
        f"{value:,}"
        for value in file_df["Missing Values"]
    ],
    padding=5
)

ax2.grid(axis="x", linestyle="--", alpha=0.3)
ax2.set_axisbelow(True)

fig2.tight_layout()

fig2.savefig(
    output_dir / "missing_values_by_file.png",
    dpi=300,
    bbox_inches="tight"
)

# Display Both Charts

print("\nTotal Records Analyzed:", f"{total_rows:,}")
print(
    "Total Missing Values:",
    f"{sum(overall_missing.values()):,}"
)

print("\nCharts saved successfully:")
print(output_dir / "missing_values_by_feature.png")
print(output_dir / "missing_values_by_file.png")

print("\nMissing Values by Feature:")
for feature, count in overall_missing.items():
    if count > 0:
        print(f"{feature}: {count:,}")

print("\nMissing Values by Original CSV File:")
for filename, count in file_missing.items():
    print(f"{filename}: {count:,}")

plt.show()
