
# Import libraries for feature selection
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

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

CHART_FOLDER = Path(
    "dataset/eda_reports/charts"
)


# Define the feature selection settings
SAMPLE_PER_CLASS = 10000
CHUNK_SIZE = 100000
RANDOM_STATE = 42
IMPORTANCE_THRESHOLD = 0.001


# Create the chart output folder
CHART_FOLDER.mkdir(parents=True, exist_ok=True)


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

class_counts = labels.value_counts()

print(f"Number of classes: {len(class_counts)}")


# Select a reproducible sample from each class
rng = np.random.default_rng(RANDOM_STATE)

sample_indices = []

for class_name, class_count in class_counts.items():

    class_indices = np.flatnonzero(
        labels.to_numpy() == class_name
    )

    sample_count = min(
        SAMPLE_PER_CLASS,
        class_count
    )

    selected_indices = rng.choice(
        class_indices,
        size=sample_count,
        replace=False
    )

    sample_indices.extend(selected_indices.tolist())


sample_indices = np.sort(
    np.asarray(sample_indices, dtype=np.int64)
)

print(f"Stratified sample rows: {len(sample_indices):,}")


# Read the selected rows from the complete dataset in chunks
sampled_chunks = []

processed_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATASET_PATH,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    chunk.columns = chunk.columns.str.strip()

    chunk_positions = np.arange(
        processed_rows,
        processed_rows + len(chunk)
    )

    mask = np.isin(
        chunk_positions,
        sample_indices
    )

    selected_chunk = chunk.loc[mask]

    if not selected_chunk.empty:
        sampled_chunks.append(selected_chunk)

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
sample_df.columns = sample_df.columns.str.strip()

X = sample_df.drop(columns=["Label"])
y = sample_df["Label"].astype(str).str.strip()


# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print(f"Rows after cleaning: {len(X):,}")
print(f"Feature columns: {len(X.columns)}")


# Convert text class labels into numeric labels
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nLabel Mapping:")

for encoded_value, class_name in enumerate(
    label_encoder.classes_
):
    print(f"{class_name} -> {encoded_value}")


# Train a preliminary XGBoost classifier
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

model.fit(X, y_encoded)

print("Preliminary XGBoost training completed.")


# Calculate feature importance scores
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance.sort_values(
    by="Importance",
    ascending=False,
    inplace=True
)

print("\nTop 20 Features:")

print(
    feature_importance.head(20).to_string(index=False)
)


# Select features based on the importance threshold
selected_df = feature_importance.loc[
    feature_importance["Importance"] >= IMPORTANCE_THRESHOLD
].copy()

removed_df = feature_importance.loc[
    feature_importance["Importance"] < IMPORTANCE_THRESHOLD
].copy()

selected_features = selected_df["Feature"].tolist()

if not selected_features:
    raise ValueError(
        "No features were selected. "
        "Review the importance threshold."
    )

print("\n" + "=" * 60)
print("Feature Selection Results")
print("=" * 60)

print(f"Importance threshold: {IMPORTANCE_THRESHOLD}")
print(f"Original features: {len(feature_importance)}")
print(f"Selected features: {len(selected_df)}")
print(f"Removed features: {len(removed_df)}")


# Save the selected feature names to JSON
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

print(f"Saved selected features to: {OUTPUT_PATH}")


# Chart 1: Top 63 Selected Important Features

selected_df = selected_df.sort_values(
    by="Importance",
    ascending=True
)

fig1, ax1 = plt.subplots(
    figsize=(13, max(12, len(selected_df) * 0.32))
)

bars1 = ax1.barh(
    selected_df["Feature"],
    selected_df["Importance"],
    color="steelblue"
)

ax1.set_xlabel("XGBoost Feature Importance Score")
ax1.set_ylabel("Feature Name")
ax1.set_title(
    f"Top {len(selected_df)} Selected Important Features"
)

ax1.bar_label(
    bars1,
    labels=[
        f"{value:.4f}"
        for value in selected_df["Importance"]
    ],
    padding=3,
    fontsize=7
)

ax1.grid(axis="x", linestyle="--", alpha=0.3)
ax1.set_axisbelow(True)

fig1.tight_layout()

fig1.savefig(
    CHART_FOLDER / "top_63_important_features.png",
    dpi=300,
    bbox_inches="tight"
)


# Chart 2: Feature Selection Threshold = 0.001

plot_df = feature_importance.sort_values(
    by="Importance",
    ascending=True
).copy()

plot_df["Selection"] = np.where(
    plot_df["Importance"] >= IMPORTANCE_THRESHOLD,
    "Selected",
    "Removed"
)

colors = plot_df["Selection"].map({
    "Selected": "seagreen",
    "Removed": "indianred"
})

fig2, ax2 = plt.subplots(
    figsize=(14, max(14, len(plot_df) * 0.30))
)

ax2.scatter(
    plot_df["Importance"],
    plot_df["Feature"],
    c=colors,
    s=45
)

ax2.axvline(
    x=IMPORTANCE_THRESHOLD,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Threshold = {IMPORTANCE_THRESHOLD}"
)

ax2.set_xlabel("XGBoost Feature Importance Score")
ax2.set_ylabel("Feature Name")

ax2.set_title(
    "Feature Selection Based on Importance Threshold"
)

ax2.grid(axis="x", linestyle="--", alpha=0.3)
ax2.set_axisbelow(True)

ax2.legend(
    title="Feature Selection",
    loc="lower right"
)

fig2.tight_layout()

fig2.savefig(
    CHART_FOLDER / "feature_selection_threshold.png",
    dpi=300,
    bbox_inches="tight"
)


# Display Both Charts
print("\nCharts generated successfully!")

print(
    CHART_FOLDER / "top_63_important_features.png"
)

print(
    CHART_FOLDER / "feature_selection_threshold.png"
)

plt.show()

print("\nFeature selection completed.")
