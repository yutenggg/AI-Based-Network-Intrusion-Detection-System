# Import required libraries
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# Set project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "processed"
MODEL_DIR = PROJECT_DIR / "model"

TRAIN_PATH = DATASET_DIR / "selected_train.csv"
VALIDATION_PATH = DATASET_DIR / "selected_validation.csv"

MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"


# Load a dataset
def load_dataset(path, feature_columns):
    print(f"\nLoading: {path.name}")

    # Define data types to reduce memory usage
    dtype_config = {
        feature: np.float32
        for feature in feature_columns
    }
    dtype_config["Label"] = str

    # Read the CSV dataset
    df = pd.read_csv(
        path,
        dtype=dtype_config,
        low_memory=False
    )

    print(f"Rows loaded: {len(df):,}")
    return df


# Train and evaluate the XGBoost model
def main():
    start_time = time.time()

    print("XGBoost Model Training")
    print("-" * 40)

    # Create the model directory if it does not exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Check whether the required dataset files exist
    for path in [TRAIN_PATH, VALIDATION_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

    # Read column names without loading the entire dataset
    columns = pd.read_csv(
        TRAIN_PATH,
        nrows=0
    ).columns.tolist()

    if "Label" not in columns:
        raise ValueError(
            "The training dataset does not contain a Label column."
        )

    # Separate feature columns from the target label
    feature_columns = [
        column for column in columns
        if column != "Label"
    ]

    print(f"\nNumber of input features: {len(feature_columns)}")

    # Load training and validation datasets
    train_df = load_dataset(
        TRAIN_PATH,
        feature_columns
    )

    validation_df = load_dataset(
        VALIDATION_PATH,
        feature_columns
    )

    # Check whether feature columns match
    validation_features = [
        column for column in validation_df.columns
        if column != "Label"
    ]

    if feature_columns != validation_features:
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    # Separate input features and target labels
    X_train = train_df[feature_columns]
    y_train_text = train_df["Label"].str.strip()

    X_validation = validation_df[feature_columns]
    y_validation_text = validation_df["Label"].str.strip()

    # Check for missing and infinite numerical values
    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError(
            "Training data contains NaN or infinite values."
        )

    if not np.isfinite(X_validation.to_numpy()).all():
        raise ValueError(
            "Validation data contains NaN or infinite values."
        )

    # Display the training label distribution
    print("\nTraining label distribution:")
    print(y_train_text.value_counts().to_string())

    # Encode text labels into numeric class IDs
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)

    # Check whether validation contains unknown labels
    unknown_labels = (
        set(y_validation_text.unique())
        - set(label_encoder.classes_)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown validation labels: {unknown_labels}"
        )

    # Encode validation labels using the training label encoder
    y_validation = label_encoder.transform(
        y_validation_text
    )

    # Get the number of classes
    num_classes = len(label_encoder.classes_)

    print(f"\nNumber of classes found: {num_classes}")
    print("Class mapping:")

    for class_id, label in enumerate(label_encoder.classes_):
        print(f"  {class_id}: {label}")

    # Release the original DataFrames to reduce memory usage
    del train_df
    del validation_df

    # Configure the XGBoost classifier
    print("\nPreparing XGBoost model...")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )

    # Train the model using the training dataset
    print("\nStarting XGBoost training...")
    print("Training may take some time.")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=10
    )

    # Evaluate the model using the validation dataset
    print("\nEvaluating validation performance...")

    predictions = model.predict(X_validation)

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Display validation results
    print("\nValidation Results")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # Display the classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=np.arange(num_classes),
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # Save the trained model and label encoder
    print("\nSaving trained model...")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # Save the mapping between numeric IDs and original labels
    label_mapping = {
        str(int(class_id)): str(label)
        for class_id, label in enumerate(
            label_encoder.classes_
        )
    }

    with open(
        LABEL_MAPPING_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            label_mapping,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Display the saved file paths
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
    print(f"Label mapping saved to: {LABEL_MAPPING_PATH}")

    # Display the total training time
    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed / 60:.2f} minutes")
    print("XGBoost training completed successfully!")


# Run the main function
if __name__ == "__main__":
    main()
