
import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from uuid import uuid4


# Define File Paths

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "xgboost_model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "label_encoder.pkl"
TEST_PATH = BASE_DIR / "dataset" / "processed" / "selected_test.csv"

API_URL = "http://127.0.0.1:5000/predict"

TEST_RECORDS = 1000


# Load Model

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

FEATURES = model.get_booster().feature_names


# Load Testing Sample

print("Loading Testing Dataset...")

df = pd.read_csv(TEST_PATH, nrows=TEST_RECORDS)

X_test = df[FEATURES].copy()

print("Testing Records:", len(X_test))
print("Testing Features:", len(FEATURES))


# Generate Local Predictions

print("\nGenerating Local Predictions...")

local_predictions = model.predict(X_test)

local_labels = label_encoder.inverse_transform(
    local_predictions.astype(int)
)

print("Local predictions completed!")


# Prepare CSV Upload

csv_data = X_test.to_csv(
    index=False
).encode("utf-8")

boundary = uuid4().hex

body = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="file"; '
    'filename="selected_test_sample.csv"\r\n'
    "Content-Type: text/csv\r\n\r\n"
).encode("utf-8") + csv_data + (
    f"\r\n--{boundary}--\r\n"
).encode("utf-8")


# Send Request to Flask API

print("\nSending CSV to Flask API...")

request = Request(
    API_URL,
    data=body,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    },
    method="POST"
)

try:
    with urlopen(request, timeout=300) as response:
        status_code = response.status
        response_data = json.loads(
            response.read().decode("utf-8")
        )

except HTTPError as error:
    print("API HTTP Error:", error.code)
    print(error.read().decode("utf-8"))
    raise SystemExit(1)

except URLError as error:
    print("Cannot connect to Flask API.")
    print("Please check whether api.py is running.")
    print("Details:", error.reason)
    raise SystemExit(1)


# Verify API Response

print("\nAPI Verification Results")
print("=" * 50)

print("HTTP Status Code:", status_code)

if status_code != 200:
    raise SystemExit("API request failed.")

if response_data.get("status") != "success":
    raise SystemExit(
        f"API returned an error: {response_data}"
    )

api_results = response_data.get("results", [])

if len(api_results) != len(X_test):
    raise SystemExit("Incorrect number of API results.")

api_labels = [
    result["attack_type"]
    for result in api_results
]

predictions_match = np.array_equal(
    local_labels,
    api_labels
)

print("API Status:", response_data["status"])
print("Total Records:", response_data["total_records"])
print("Normal Records:", response_data["normal_records"])
print("Detected Attacks:", response_data["detected_attacks"])
print("Results Returned:", len(api_results))

print(
    "Prediction Matches Local Model:",
    predictions_match
)


# Verify Detection Summary

summary = response_data.get("attack_summary", {})

summary_total = sum(summary.values())

summary_valid = (
    summary_total == len(X_test)
    and response_data["normal_records"] == summary.get("BENIGN", 0)
    and response_data["detected_attacks"]
    == len(X_test) - summary.get("BENIGN", 0)
)

print("Detection Summary Valid:", summary_valid)

print("\nAttack Summary:")

for attack, count in summary.items():
    print(f"{attack}: {count}")


# Verify Detection Result Fields

required_fields = [
    "row",
    "attack_type",
    "risk_level",
    "confidence_score",
    "recommendation"
]

fields_valid = all(
    all(field in result for field in required_fields)
    for result in api_results
)

print("\nDetection Result Fields Valid:", fields_valid)


# Final Result

print("\n" + "=" * 50)

if predictions_match and summary_valid and fields_valid:
    print("API Verification: PASSED")
else:
    print("API Verification: FAILED")
