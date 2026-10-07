import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("dataset/processed/cleaned_CICIDS2017.csv")
FEATURE_PATH = Path("training/selected_feature.json")

TRAIN_OUTPUT = Path("dataset/processed/selected_train.csv")
VALIDATION_OUTPUT = Path("dataset/processed/selected_validation.csv")
TEST_OUTPUT = Path("dataset/processed/selected_test.csv")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.20

RANDOM_STATE = 42

with open(FEATURE_PATH, "r") as f:
    selected_features = json.load(f)["features"]

print("=" * 60)
print("Selected Features")
print("=" * 60)

print(f"Number of selected features: {len(selected_features)}")

for feature in selected_features:
    print(feature)


# ==============================
# Read Label column
# ==============================

print("\n" + "=" * 60)
print("Reading labels")
print("=" * 60)

labels = pd.read_csv(
    DATASET_PATH,
    usecols=["Label"]
)["Label"].str.strip()

print(f"Total rows: {len(labels):,}")


# ==============================
# Create stratified indices
# ==============================

print("\n" + "=" * 60)
print("Creating stratified split")
print("=" * 60)


all_indices = pd.Series(range(len(labels)))

train_indices, temp_indices = train_test_split(
    all_indices,
    test_size=(VALIDATION_RATIO + TEST_RATIO),
    stratify=labels,
    random_state=RANDOM_STATE
)

temp_labels = labels.iloc[temp_indices]

validation_ratio_adjusted = (
    VALIDATION_RATIO /
    (VALIDATION_RATIO + TEST_RATIO)
)

validation_indices, test_indices = train_test_split(
    temp_indices,
    test_size=(TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO)),
    stratify=temp_labels,
    random_state=RANDOM_STATE
)


# Convert to sets for fast lookup

train_indices = set(train_indices)
validation_indices = set(validation_indices)
test_indices = set(test_indices)


print(f"Train rows:      {len(train_indices):,}")
print(f"Validation rows: {len(validation_indices):,}")
print(f"Test rows:       {len(test_indices):,}")


# ==============================
# Prepare output columns
# ==============================

output_columns = selected_features + ["Label"]


# ==============================
# Remove old output files
# ==============================

for output_file in [
    TRAIN_OUTPUT,
    VALIDATION_OUTPUT,
    TEST_OUTPUT
]:

    if output_file.exists():
        output_file.unlink()


# ==============================
# Process dataset in chunks
# ==============================

print("\n" + "=" * 60)
print("Processing dataset in chunks")
print("=" * 60)

CHUNK_SIZE = 100000

first_train = True
first_validation = True
first_test = True

processed_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATASET_PATH,
        usecols=output_columns,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    chunk.index = range(
        processed_rows,
        processed_rows + len(chunk)
    )

    processed_rows += len(chunk)

    train_mask = chunk.index.isin(train_indices)
    validation_mask = chunk.index.isin(validation_indices)
    test_mask = chunk.index.isin(test_indices)

    train_chunk = chunk.loc[train_mask]
    validation_chunk = chunk.loc[validation_mask]
    test_chunk = chunk.loc[test_mask]

    if len(train_chunk) > 0:
        train_chunk.to_csv(
            TRAIN_OUTPUT,
            mode="w" if first_train else "a",
            header=first_train,
            index=False
        )
        first_train = False

    if len(validation_chunk) > 0:
        validation_chunk.to_csv(
            VALIDATION_OUTPUT,
            mode="w" if first_validation else "a",
            header=first_validation,
            index=False
        )
        first_validation = False

    if len(test_chunk) > 0:
        test_chunk.to_csv(
            TEST_OUTPUT,
            mode="w" if first_test else "a",
            header=first_test,
            index=False
        )
        first_test = False

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows processed: {processed_rows:,}"
    )


# ==============================
# Final verification
# ==============================

print("\n" + "=" * 60)
print("Split completed")
print("=" * 60)

print(f"Train file:      {TRAIN_OUTPUT}")
print(f"Validation file: {VALIDATION_OUTPUT}")
print(f"Test file:       {TEST_OUTPUT}")

print("\nExpected row counts:")
print(f"Train:      {len(train_indices):,}")
print(f"Validation: {len(validation_indices):,}")
print(f"Test:       {len(test_indices):,}")

print("\nDone!")
