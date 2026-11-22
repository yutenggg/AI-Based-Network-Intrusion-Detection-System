
# Import libraries
import json
import joblib
import pandas as pd
import urllib.request
import urllib.error

from io import BytesIO
from pathlib import Path


# Define file paths
DATASET_PATH = Path("dataset/processed/selected_test.csv")
MODEL_PATH = Path("model/xgboost_model.pkl")
ENCODER_PATH = Path("model/label_encoder.pkl")

API_URL = "http://127.0.0.1:5000/predict"

SAMPLES_PER_CLASS = 20


# Load Model and Label Encoder
print("Loading XGBoost Model...")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

features = model.get_booster().feature_names

print("Model loaded successfully!")
print("Number of Features:", len(features))


# Load Testing Dataset
print("\nLoading Testing Dataset...")

df = pd.read_csv(DATASET_PATH)

if "Label" not in df.columns:
    raise ValueError("Ground Truth Label column is missing.")

# Select up to 20 records from each attack class
sample_df = (
    df.groupby("Label", sort=False, group_keys=False)
    .head(SAMPLES_PER_CLASS)
    .reset_index(drop=True)
)

X = sample_df[features].copy()

true_labels = sample_df["Label"].astype(str).tolist()

print("Selected Records:", len(sample_df))
print("Number of Ground Truth Classes:", sample_df["Label"].nunique())


# Generate Local Predictions
print("\nGenerating Local Predictions...")

local_predictions = model.predict(X)
local_predictions = label_encoder.inverse_transform(
    local_predictions.astype(int)
)

local_predictions = [str(label) for label in local_predictions]

print("Local predictions completed!")


# Upload Testing Data to Flask API
print("\nSending CSV to Flask API...")

csv_buffer = BytesIO()

X.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

boundary = "----AIIDSBoundary123456"

body = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="file"; filename="attack_testing.csv"\r\n'
    "Content-Type: text/csv\r\n\r\n"
).encode("utf-8")

body += csv_buffer.getvalue()
body += f"\r\n--{boundary}--\r\n".encode("utf-8")

request = urllib.request.Request(
    API_URL,
    data=body,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    },
    method="POST"
)

try:
    with urllib.request.urlopen(request, timeout=120) as response:
        status_code = response.status
        api_data = json.loads(response.read().decode("utf-8"))

except urllib.error.HTTPError as error:
    print("API returned an HTTP error:", error.code)
    print(error.read().decode("utf-8"))
    raise

except Exception as error:
    print("Cannot connect to Flask API.")
    print("Details:", str(error))
    raise


# Verify API Results
print("\nAPI Attack Detection Results")
print("=" * 50)

api_predictions = [
    str(result["attack_type"])
    for result in api_data["results"]
]

prediction_matches = api_predictions == local_predictions

results_count_valid = (
    len(api_data["results"]) == len(sample_df)
)

fields = [
    "attack_type",
    "risk_level",
    "confidence_score",
    "recommendation"
]

result_fields_valid = all(
    all(field in result and result[field] is not None
        for field in fields)
    for result in api_data["results"]
)

attack_summary = api_data["attack_summary"]

summary_valid = (
    isinstance(attack_summary, dict)
    and sum(attack_summary.values()) == len(sample_df)
)

ground_truth_accuracy = sum(
    predicted == actual
    for predicted, actual in zip(api_predictions, true_labels)
) / len(true_labels)


print("HTTP Status Code:", status_code)
print("API Status:", api_data.get("status"))
print("Total Records:", api_data.get("total_records"))
print("Normal Records:", api_data.get("normal_records"))
print("Detected Attacks:", api_data.get("detected_attacks"))
print("Results Returned:", len(api_data["results"]))

print("Prediction Matches Local Model:", prediction_matches)
print("Result Count Valid:", results_count_valid)
print("Detection Summary Valid:", summary_valid)
print("Detection Result Fields Valid:", result_fields_valid)

print(
    "Sample Ground Truth Accuracy:",
    f"{ground_truth_accuracy:.4f}"
)


# Display Attack Summary
print("\nAttack Summary:")

for attack, count in attack_summary.items():
    print(f"{attack}: {count}")


# Display Final Verification
print("\n" + "=" * 50)

all_passed = (
    status_code == 200
    and api_data.get("status") == "success"
    and results_count_valid
    and prediction_matches
    and summary_valid
    and result_fields_valid
)

if all_passed:
    print("API Attack Detection Verification: PASSED")
else:
    print("API Attack Detection Verification: FAILED")
