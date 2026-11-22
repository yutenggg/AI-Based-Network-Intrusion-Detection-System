
import json
import joblib
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_PATH = BASE_DIR / "training" / "selected_feature.json"
MODEL_PATH = BASE_DIR / "model" / "xgboost_model.pkl"
DATASET_DIR = BASE_DIR / "dataset" / "processed"

DATASET_FILES = [
    "selected_train.csv",
    "selected_validation.csv",
    "selected_test.csv"
]


# Load Selected Features

with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    selected_features = json.load(file)["features"]


# Load Trained Model

model = joblib.load(MODEL_PATH)
model_features = model.get_booster().feature_names

if model_features is None:
    raise ValueError("Model feature names are unavailable.")


print("\nDataset Splitting Verification")
print("=" * 50)

print("JSON Features:", len(selected_features))
print("Model Features:", len(model_features))
print(
    "Model Matches JSON:",
    set(model_features) == set(selected_features)
)

if set(model_features) != set(selected_features):
    raise ValueError("Model features do not match selected features.")


# Verify Dataset Columns

training_features = None
all_valid = True

for filename in DATASET_FILES:

    file_path = DATASET_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    columns = pd.read_csv(file_path, nrows=0).columns.tolist()

    features = [
        column for column in columns
        if column != "Label"
    ]

    matches_json = set(features) == set(selected_features)
    matches_model = features == model_features

    if training_features is None:
        training_features = features

    matches_training = features == training_features

    valid = (
        len(features) == 63
        and matches_json
        and matches_model
        and matches_training
        and columns.count("Label") == 1
    )

    all_valid = all_valid and valid

    print("\n" + filename)
    print("Feature Count:", len(features))
    print("Matches JSON:", matches_json)
    print("Matches Model Order:", matches_model)
    print("Matches Training Order:", matches_training)
    print("Verification:", "PASS" if valid else "FAIL")


# Final Result

print("\n" + "=" * 50)

if all_valid:
    print("Dataset Splitting Verification: PASSED")
else:
    print("Dataset Splitting Verification: FAILED")
