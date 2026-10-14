# Import libraries for dataset splitting
import pandas as pd
import json

from pathlib import Path
from sklearn.model_selection import train_test_split


# Define the input dataset and selected feature file
DATASET_PATH = Path(
    "dataset/processed/cleaned_CICIDS2017.csv"
)

FEATURE_PATH = Path(
    "training/selected_feature.json"
)


# Define the output files for the three datasets
TRAIN_OUTPUT = Path(
    "dataset/processed/selected_train.csv"
)

VALIDATION_OUTPUT = Path(
    "dataset/processed/selected_validation.csv"
)

TEST_OUTPUT = Path(
    "dataset/processed/selected_test.csv"
)


# Define the dataset split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.20

# Use a fixed random seed for reproducible splitting
RANDOM_STATE = 42

# Process the dataset in smaller batches
CHUNK_SIZE = 100000


# Check that the required input files exist
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATASET_PATH}"
    )

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Selected feature file not found: {FEATURE_PATH}"
    )


# Check that the split ratios add up to 100%
if not abs(
    TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0
) < 1e-9:
    raise ValueError(
        "Train, validation, and test ratios must add up to 1."
    )


print("=" * 60)
print("CICIDS2017 Dataset Splitting")
print("=" * 60)


# Load the selected feature names from the JSON file
with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    selected_features = json.load(file)["features"]

print(
    f"Number of selected features: {len(selected_features)}"
)


# Read only the Label column to determine the class of each row
print("\nReading dataset labels...")

labels = pd.read_csv(
    DATASET_PATH,
    usecols=["Label"]
)["Label"].str.strip()

print(f"Total rows: {len(labels):,}")


# Create row positions for the complete dataset
all_indices = pd.Series(range(len(labels)))


# Split the dataset into training and temporary groups
# The temporary group will later be divided into validation and test
print("\nCreating stratified train, validation, and test splits...")

train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=VALIDATION_RATIO + TEST_RATIO,
    stratify=labels,
    random_state=RANDOM_STATE
)


# Obtain the labels belonging to the temporary group
temp_labels = labels.iloc[temp_indices]


# Divide the temporary group into validation and test
# Validation = 10% of the complete dataset
# Test = 20% of the complete dataset
validation_indices, test_indices = train_test_split(
    temp_indices,
    test_size=TEST_RATIO / (
        VALIDATION_RATIO + TEST_RATIO
    ),
    stratify=temp_labels,
    random_state=RANDOM_STATE
)


# Convert row positions to sets for membership checking
train_indices = set(train_indices)
validation_indices = set(validation_indices)
test_indices = set(test_indices)


# Display the expected size of each split
print("\nExpected split sizes:")

print(f"Train:      {len(train_indices):,} rows")
print(f"Validation: {len(validation_indices):,} rows")
print(f"Test:       {len(test_indices):,} rows")


# Keep only the selected input features and the target label
# The final number of columns should be selected features + Label
output_columns = selected_features + ["Label"]


# Remove old output files before creating the new split files
for output_file in [
    TRAIN_OUTPUT,
    VALIDATION_OUTPUT,
    TEST_OUTPUT
]:
    if output_file.exists():
        output_file.unlink()


# Create the output directory if necessary
TRAIN_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# Track whether each output file needs its header
first_train = True
first_validation = True
first_test = True

# Track the total number of rows processed
processed_rows = 0


# Read and split the dataset in chunks to reduce memory usage
print("\nProcessing dataset in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATASET_PATH,
        usecols=output_columns,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    # Assign the original row positions to the current chunk
    chunk.index = range(
        processed_rows,
        processed_rows + len(chunk)
    )

    # Update the number of rows processed
    processed_rows += len(chunk)

    # Identify rows belonging to each split
    train_mask = chunk.index.isin(train_indices)
    validation_mask = chunk.index.isin(validation_indices)
    test_mask = chunk.index.isin(test_indices)

    # Extract the corresponding rows
    train_chunk = chunk.loc[train_mask]
    validation_chunk = chunk.loc[validation_mask]
    test_chunk = chunk.loc[test_mask]

    # Append training rows to the training CSV
    if not train_chunk.empty:
        train_chunk.to_csv(
            TRAIN_OUTPUT,
            mode="w" if first_train else "a",
            header=first_train,
            index=False
        )

        first_train = False

    # Append validation rows to the validation CSV
    if not validation_chunk.empty:
        validation_chunk.to_csv(
            VALIDATION_OUTPUT,
            mode="w" if first_validation else "a",
            header=first_validation,
            index=False
        )

        first_validation = False

    # Append test rows to the test CSV
    if not test_chunk.empty:
        test_chunk.to_csv(
            TEST_OUTPUT,
            mode="w" if first_test else "a",
            header=first_test,
            index=False
        )

        first_test = False

    # Display progress after processing each chunk
    print(
        f"Processed chunk {chunk_number} | "
        f"Rows processed: {processed_rows:,}"
    )


# Display the final output locations and expected row counts
print("\n" + "=" * 60)
print("Dataset Splitting Completed")
print("=" * 60)

print(f"Train file:      {TRAIN_OUTPUT}")
print(f"Validation file: {VALIDATION_OUTPUT}")
print(f"Test file:       {TEST_OUTPUT}")

print("\nExpected row counts:")
print(f"Train:      {len(train_indices):,}")
print(f"Validation: {len(validation_indices):,}")
print(f"Test:       {len(test_indices):,}")

print("\nDone!")