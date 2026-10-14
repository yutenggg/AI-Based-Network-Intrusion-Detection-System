# Import libraries for feature selection
import pandas as pd
import numpy as np
import json

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Define the cleaned dataset and feature selection output paths
DATASET_PATH = Path(
    "dataset/processed/cleaned_CICIDS2017.csv"
)

OUTPUT_PATH = Path(
    "training/selected_feature.json"
)


# Define the feature selection settings
# Limit the sample size for each class to reduce memory usage
SAMPLE_PER_CLASS = 10000

# Read the large dataset in smaller batches
CHUNK_SIZE = 100000

# Set a fixed random seed so sampling can be reproduced
RANDOM_STATE = 42

# Set the minimum importance score for retaining a feature
IMPORTANCE_THRESHOLD = 0.001


# Check whether the cleaned dataset exists
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATASET_PATH}"
    )


print("=" * 60)
print("CICIDS2017 Feature Selection")
print("=" * 60)


# Read only the Label column to identify the available classes
labels = pd.read_csv(
    DATASET_PATH,
    usecols=["Label"]
)["Label"].str.strip()

print(f"Total rows: {len(labels):,}")

# Get the number of records belonging to each class
class_counts = labels.value_counts()

print(f"Number of classes: {len(class_counts)}")


# Select a reproducible sample from each class
# Rare classes will contribute all available records
# when they contain fewer rows than SAMPLE_PER_CLASS
rng = np.random.default_rng(RANDOM_STATE)

sample_indices = []

for class_name, class_count in class_counts.items():

    # Find the row positions belonging to the current class
    class_indices = np.flatnonzero(
        labels.to_numpy() == class_name
    )

    # Determine how many rows to sample from this class
    sample_count = min(
        SAMPLE_PER_CLASS,
        class_count
    )

    # Randomly select row positions without replacement
    selected_indices = rng.choice(
        class_indices,
        size=sample_count,
        replace=False
    )

    sample_indices.extend(selected_indices.tolist())


# Sort the selected positions for easier chunk processing
sample_indices = np.sort(
    np.asarray(sample_indices, dtype=np.int64)
)

print(f"Stratified sample rows: {len(sample_indices):,}")


# Read the selected rows from the complete dataset in chunks
# This avoids loading all 2.5 million rows into memory at once
sampled_chunks = []

selected_position_set = set(sample_indices.tolist())

processed_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATASET_PATH,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    # Standardise column names
    chunk.columns = chunk.columns.str.strip()

    # Assign original row positions to the current chunk
    chunk_positions = np.arange(
        processed_rows,
        processed_rows + len(chunk)
    )

    # Identify rows selected for the stratified sample
    mask = np.isin(
        chunk_positions,
        sample_indices
    )

    # Keep only the selected sample rows
    selected_chunk = chunk.loc[mask]

    if not selected_chunk.empty:
        sampled_chunks.append(selected_chunk)

    # Update the number of rows already processed
    processed_rows += len(chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows scanned: {processed_rows:,}"
    )


# Combine the sampled rows into one dataframe
sample_df = pd.concat(
    sampled_chunks,
    ignore_index=True
)

print(f"\nSample rows loaded: {len(sample_df):,}")
print(f"Sample columns: {len(sample_df.columns)}")


# Separate the input features and target labels
# Label is the target and must not be included as an input feature
sample_df.columns = sample_df.columns.str.strip()

X = sample_df.drop(columns=["Label"])
y = sample_df["Label"].astype(str).str.strip()

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Keep only rows with valid feature values and labels
valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print(f"Rows after cleaning: {len(X):,}")
print(f"Feature columns: {len(X.columns)}")


# Convert text class labels into numeric labels for XGBoost
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# Display the mapping between original and encoded labels
print("\nLabel Mapping:")

for encoded_value, class_name in enumerate(
    label_encoder.classes_
):
    print(f"{class_name} -> {encoded_value}")


# Create a preliminary XGBoost classifier
# This model estimates feature importance for feature selection
print("\n" + "=" * 60)
print("Training Preliminary XGBoost Model")
print("=" * 60)

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=2
)

# Train the preliminary model on the sampled data
model.fit(X, y_encoded)

print("Preliminary XGBoost training completed.")


# Obtain the importance score of each input feature
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

# Sort features from most important to least important
feature_importance.sort_values(
    by="Importance",
    ascending=False,
    inplace=True
)

print("\nTop 20 Features:")

print(
    feature_importance.head(20).to_string(index=False)
)


# Retain features that meet the selected importance threshold
selected_features = feature_importance.loc[
    feature_importance["Importance"] >= IMPORTANCE_THRESHOLD,
    "Feature"
].tolist()

# Stop if no features meet the threshold
if not selected_features:
    raise ValueError(
        "No features were selected. "
        "Review the importance threshold."
    )

print(
    f"\nImportance threshold: {IMPORTANCE_THRESHOLD}"
)

print(
    f"Selected features count: {len(selected_features)}"
)


# Save the selected feature names to a JSON file
# Training, validation, testing, and website input must
# reuse this same feature list
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        {"features": selected_features},
        file,
        indent=4
    )


# Display the final output location
print(f"\nSaved selected features to: {OUTPUT_PATH}")
print("Feature selection completed.")