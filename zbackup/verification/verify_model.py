
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Define File Paths

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "xgboost_model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "label_encoder.pkl"
TEST_PATH = BASE_DIR / "dataset" / "processed" / "selected_test.csv"

OUTPUT_DIR = BASE_DIR / "dataset" / "eda_reports"
CHART_DIR = OUTPUT_DIR / "charts"


# Load Model and Label Encoder

print("Loading XGBoost Model...")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

model_features = model.get_booster().feature_names
class_names = label_encoder.classes_

if model_features is None:
    raise ValueError("Model feature names are unavailable.")

print("Model loaded successfully!")
print("Number of Features:", len(model_features))
print("Number of Classes:", len(class_names))


# Load Testing Dataset

print("\nLoading Testing Dataset...")

df = pd.read_csv(TEST_PATH)

if "Label" not in df.columns:
    raise ValueError("Testing Dataset does not contain Label.")

missing_features = [
    feature for feature in model_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

X_test = df[model_features].copy()
y_test = df["Label"].astype(str).str.strip()

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    raise ValueError("Testing Dataset contains invalid values.")

if not set(y_test.unique()).issubset(set(class_names)):
    raise ValueError("Testing Dataset contains unknown labels.")

print("Testing Records:", len(df))
print("Testing Features:", len(X_test.columns))


# Generate Model Predictions

print("\nGenerating Predictions...")

predictions = model.predict(X_test)

y_pred = label_encoder.inverse_transform(
    predictions.astype(int)
)

print("Prediction completed successfully!")


# Calculate Performance Metrics

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    labels=class_names,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    labels=class_names,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    labels=class_names,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    labels=class_names,
    average="weighted",
    zero_division=0
)


# Display Performance Metrics

print("\nModel Verification Results")
print("=" * 50)

print(f"Accuracy:       {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall:    {recall:.4f}")
print(f"Macro F1-Score:  {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")

print("=" * 50)


# Generate Classification Report

report = classification_report(
    y_test,
    y_pred,
    labels=class_names,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

report_path = OUTPUT_DIR / "model_classification_report.csv"

report_df.to_csv(report_path)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0
    )
)

print("Classification Report saved:", report_path)


# Generate Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names,
    normalize="true"
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(16, 13))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    values_format=".2f",
    colorbar=True
)

ax.set_title("XGBoost Model - Normalized Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

plt.tight_layout()

chart_path = CHART_DIR / "model_confusion_matrix.png"

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion Matrix saved:", chart_path)


# Save Performance Metrics

metrics = {
    "Testing Records": int(len(df)),
    "Number of Features": int(len(model_features)),
    "Number of Classes": int(len(class_names)),
    "Accuracy": float(accuracy),
    "Macro Precision": float(precision),
    "Macro Recall": float(recall),
    "Macro F1-Score": float(macro_f1),
    "Weighted F1-Score": float(weighted_f1)
}

metrics_path = OUTPUT_DIR / "model_verification_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("Performance Metrics saved:", metrics_path)

print("\nModel Verification Completed Successfully!")
