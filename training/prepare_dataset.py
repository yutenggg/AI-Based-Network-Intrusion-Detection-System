# Import libraries for dataset processing
import pandas as pd
import numpy as np
from pathlib import Path


# Define the folder containing the original CICIDS2017 CSV files
DATASET_FOLDER = Path("dataset/MachineLearningCVE")

# Define the output folder and cleaned dataset filename
OUTPUT_FOLDER = Path("dataset/processed")
OUTPUT_FILE = OUTPUT_FOLDER / "cleaned_CICIDS2017.csv"


# Create the output folder if it does not exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Find all CSV files in the dataset folder
csv_files = sorted(DATASET_FOLDER.glob("*.csv"))

# Stop if no CSV files are found
if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {DATASET_FOLDER}"
    )

print("=" * 60)
print("CICIDS2017 Dataset Preparation")
print("=" * 60)

print(f"Number of CSV files found: {len(csv_files)}")

# Store the cleaned data from each CSV file
cleaned_dataframes = []

# Process each original CSV file
for csv_file in csv_files:

    print("\n" + "-" * 60)
    print(f"Processing: {csv_file.name}")
    print("-" * 60)

    # Read the original CSV file
    df = pd.read_csv(csv_file, low_memory=False)

    # Remove unnecessary spaces from column names
    df.columns = df.columns.str.strip()

    # Record the original number of rows
    original_rows = len(df)

    # Count missing values before cleaning
    missing_count = int(df.isna().sum().sum())

    # Count positive and negative infinite values
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    infinite_count = int(
        np.isinf(df[numeric_columns]).sum().sum()
    )

    # Replace infinite values with missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows containing missing values
    df.dropna(inplace=True)

    # Remove duplicate rows within the current CSV file
    rows_before_duplicates = len(df)
    df.drop_duplicates(inplace=True)

    # Calculate the number of duplicate rows removed
    duplicate_count = rows_before_duplicates - len(df)

    # Display the cleaning results
    print(f"Original rows: {original_rows:,}")
    print(f"Missing values: {missing_count:,}")
    print(f"Infinite values: {infinite_count:,}")
    print(f"Duplicate rows removed: {duplicate_count:,}")
    print(f"Cleaned rows: {len(df):,}")

    # Keep the cleaned dataframe for merging
    cleaned_dataframes.append(df)


# Combine the cleaned CSV files into one dataset
print("\n" + "=" * 60)
print("Merging Cleaned CSV Files")
print("=" * 60)

combined_df = pd.concat(
    cleaned_dataframes,
    ignore_index=True
)

# Remove duplicates that may exist across different CSV files
rows_before_merge_deduplication = len(combined_df)

combined_df.drop_duplicates(inplace=True)
combined_df.reset_index(drop=True, inplace=True)

merged_duplicates = (
    rows_before_merge_deduplication - len(combined_df)
)

# Save the final cleaned dataset
combined_df.to_csv(OUTPUT_FILE, index=False)

# Display the final dataset information
print("\n" + "=" * 60)
print("Dataset Preparation Completed")
print("=" * 60)

print(f"Total rows before final deduplication: "
      f"{rows_before_merge_deduplication:,}")

print(f"Duplicates removed after merging: {merged_duplicates:,}")

print(f"Final rows: {len(combined_df):,}")
print(f"Total columns: {len(combined_df.columns)}")
print(f"Saved dataset: {OUTPUT_FILE}")
print("Done!")