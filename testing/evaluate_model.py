# Import required libraries
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# Set project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "processed"
MODEL_DIR = PROJECT_DIR / "model"
TESTING_DIR = PROJECT_DIR / "testing"

TEST_PATH = DATASET_DIR / "selected_test.csv"

MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

RESULT_PATH = TESTING_DIR / "test_results.txt"


# Evaluate the trained model using the independent test dataset
def main():
    print("XGBoost Independent Test Evaluation")
    print("=" * 50)

    # Check whether the required files exist
    required_files = [
        TEST_PATH,
        MODEL_PATH,
        LABEL_ENCODER_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # Load the previously trained model and label encoder
    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    print("Model loaded successfully.")
    print(
        f"Number of classes: {len(label_encoder.classes_)}"
    )

    # Read column names before loading the test dataset
    columns = pd.read_csv(
        TEST_PATH,
        nrows=0
    ).columns.tolist()

    if "Label" not in columns:
        raise ValueError(
            "The test dataset does not contain a Label column."
        )

    # Separate input features from the target label
    feature_columns = [
        column for column in columns
        if column != "Label"
    ]

    print(
        f"Number of input features: {len(feature_columns)}"
    )

    # Check the number of features expected by the model
    if len(feature_columns) != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: test dataset has "
            f"{len(feature_columns)} features, but the model "
            f"expects {model.n_features_in_}."
        )

    # Load the independent test dataset
    print("\nLoading independent test dataset...")

    dtype_config = {
        feature: np.float32
        for feature in feature_columns
    }
    dtype_config["Label"] = str

    test_df = pd.read_csv(
        TEST_PATH,
        dtype=dtype_config,
        low_memory=False
    )

    print(f"Rows loaded: {len(test_df):,}")

    # Separate test features and actual labels
    X_test = test_df[feature_columns]
    y_test_text = test_df["Label"].str.strip()

    # Check for missing or infinite feature values
    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError(
            "The test dataset contains NaN or infinite values."
        )

    # Check whether the test labels exist in the trained encoder
    unknown_labels = (
        set(y_test_text.unique())
        - set(label_encoder.classes_)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown test labels: {unknown_labels}"
        )

    # Convert actual labels to numeric class IDs
    y_test = label_encoder.transform(y_test_text)

    # Release the original DataFrame to reduce memory usage
    del test_df

    # Generate predictions using the saved model
    print("\nPredicting test dataset...")
    print("This may take some time.")

    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Generate the classification report
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )

    # Prepare the evaluation results
    results = (
        "XGBoost Independent Test Evaluation\n"
        + "=" * 50 + "\n"
        + f"Test samples: {len(y_test):,}\n"
        + f"Input features: {len(feature_columns)}\n"
        + f"Number of classes: {len(label_encoder.classes_)}\n\n"
        + f"Accuracy: {accuracy:.4f}\n"
        + f"Macro F1: {macro_f1:.4f}\n"
        + f"Weighted F1: {weighted_f1:.4f}\n\n"
        + "Classification Report\n"
        + "-" * 50 + "\n"
        + report
    )

    # Display the evaluation results
    print("\n")
    print(results)

    # Save the evaluation results to a text file
    TESTING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(results)

    print(f"\nResults saved to: {RESULT_PATH}")
    print("\nIndependent test evaluation completed.")


# Run the main function
if __name__ == "__main__":
    main()
