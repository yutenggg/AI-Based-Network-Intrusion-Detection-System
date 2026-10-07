import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
import json

DATASET_PATH = Path("dataset/processed/cleaned_CICIDS2017.csv")
OUTPUT_PATH = Path("training/selected_feature.json")

SAMPLE_PER_CLASS = 10000
CHUNK_SIZE = 100000
RANDOM_STATE = 42

print("Reading Label column...")

label_df = pd.read_csv(
    DATASET_PATH,
    usecols=["Label"]
)

print("Total rows:", len(label_df))

print("\nOriginal class distribution:")

print(
    label_df["Label"].value_counts()
)

print("\nCreating stratified sample indices...")

rng = np.random.RandomState(
    RANDOM_STATE
)

sample_indices = []

for label in label_df["Label"].unique():

    indices = label_df.index[
        label_df["Label"] == label
    ].to_numpy()

    sample_count = min(
        SAMPLE_PER_CLASS,
        len(indices)
    )

    selected = rng.choice(
        indices,
        size=sample_count,
        replace=False
    )

    sample_indices.extend(
        selected
    )

sample_indices = np.array(
    sorted(sample_indices)
)

print(
    "Total sampled rows:",
    len(sample_indices)
)

print("\nReading selected rows in chunks...")

sample_parts = []

sample_index_set = set(
    sample_indices
)

current_start = 0

for chunk in pd.read_csv(
    DATASET_PATH,
    chunksize=CHUNK_SIZE
):

    current_end = (
        current_start + len(chunk)
    )

    chunk_indices = np.arange(
        current_start,
        current_end
    )

    selected_in_chunk = np.intersect1d(
        chunk_indices,
        sample_indices
    )

    if len(selected_in_chunk) > 0:

        relative_indices = (
            selected_in_chunk
            - current_start
        )

        selected_chunk = chunk.iloc[
            relative_indices
        ]

        sample_parts.append(
            selected_chunk
        )

    current_start = current_end

    print(
        f"Processed rows: {current_end}"
    )

sample_df = pd.concat(
    sample_parts,
    ignore_index=True
)

print(
    "\nSample dataset rows:",
    len(sample_df)
)

print(
    "Sample dataset columns:",
    len(sample_df.columns)
)

sample_df.columns = (
    sample_df.columns.str.strip()
)

X = sample_df.drop(
    columns=["Label"]
)

y = sample_df["Label"]

print(
    "\nFeature columns:",
    X.shape[1]
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows containing missing values

valid_rows = X.notna().all(
    axis=1
)

X = X.loc[
    valid_rows
]

y = y.loc[
    valid_rows
]

print(
    "Rows after cleaning:",
    len(X)
)

# ==========================================
# 11. Encode labels
# ==========================================

labels = sorted(
    y.unique()
)

label_mapping = {
    label: index
    for index, label in enumerate(labels)
}

y = y.map(
    label_mapping
)

print(
    "\nNumber of classes:",
    len(label_mapping)
)

print(
    "\nLabel mapping:"
)

for label, number in label_mapping.items():

    print(
        f"{label} -> {number}"
    )

print(
    "\nTraining preliminary XGBoost model..."
)

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

model.fit(
    X,
    y
)

print(
    "Preliminary XGBoost training completed."
)

importance = (
    model.feature_importances_
)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print(
    "\nFeature Importance Ranking:"
)

print(
    feature_importance.to_string(
        index=False
    )
)

threshold = 0.001

selected_features = (
    feature_importance[
        feature_importance["Importance"]
        >= threshold
    ]["Feature"]
    .tolist()
)

print(
    "\nSelected features:",
    len(selected_features)
)

for feature in selected_features:

    print(
        feature
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w"
) as file:

    json.dump(
        {
            "features":
                selected_features
        },
        file,
        indent=4
    )

print(
    "\nSelected features saved to:"
)

print(
    OUTPUT_PATH
)