
# Import libraries
import json
import joblib
import numpy as np
import pandas as pd
import xlsxwriter
import os
import tempfile
import uuid

from collections import Counter
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Initialize Flask
app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

BATCH_SIZE = 50000
RECORDS_PER_PAGE = 20

# Store detection sessions in memory
DETECTION_SESSIONS = {}


# Define Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "xgboost_model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "label_encoder.pkl"
MAPPING_PATH = BASE_DIR / "model" / "label_mapping.json"
FEATURE_PATH = BASE_DIR / "training" / "selected_feature.json"


# Load Model Files
print("Loading AI-Based Network Intrusion Detection System...")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

with open(MAPPING_PATH, "r", encoding="utf-8") as file:
    label_mapping = json.load(file)

with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    feature_data = json.load(file)


# Get the Features
SELECTED_FEATURES = feature_data["features"]
MODEL_FEATURES = model.get_booster().feature_names

if set(SELECTED_FEATURES) != set(MODEL_FEATURES):
    raise ValueError("Selected features do not match model features.")

FEATURES = MODEL_FEATURES

print("Model loaded successfully!")
print("Number of features:", len(FEATURES))
print("Number of attack classes:", len(label_encoder.classes_))


# Get Actual Web Attack Class Names
CLASS_NAMES = [
    str(label) for label in label_encoder.classes_
]

WEB_BRUTE_FORCE = next(
    (name for name in CLASS_NAMES if "Brute Force" in name),
    "Web Attack - Brute Force"
)

WEB_SQL_INJECTION = next(
    (name for name in CLASS_NAMES if "Sql Injection" in name),
    "Web Attack - Sql Injection"
)

WEB_XSS = next(
    (name for name in CLASS_NAMES if "XSS" in name),
    "Web Attack - XSS"
)


# Define Risk Levels
RISK_LEVELS = {
    "BENIGN": "Low",
    "Bot": "High",
    "DDoS": "Critical",
    "DoS GoldenEye": "Critical",
    "DoS Hulk": "Critical",
    "DoS Slowhttptest": "High",
    "DoS slowloris": "High",
    "FTP-Patator": "High",
    "Heartbleed": "Critical",
    "Infiltration": "Critical",
    "PortScan": "Medium",
    "SSH-Patator": "High",
    WEB_BRUTE_FORCE: "High",
    WEB_SQL_INJECTION: "Critical",
    WEB_XSS: "High"
}


# Define Security Recommendations
RECOMMENDATIONS = {
    "BENIGN": "No immediate action required. Continue monitoring network traffic.",
    "Bot": "Investigate the affected device and check for botnet activity.",
    "DDoS": "Apply traffic filtering and rate limiting to reduce excessive traffic.",
    "DoS GoldenEye": "Investigate abnormal connection requests and block suspicious sources.",
    "DoS Hulk": "Apply rate limiting and block suspicious traffic sources.",
    "DoS Slowhttptest": "Review suspicious slow connections and configure connection timeouts.",
    "DoS slowloris": "Limit incomplete connections and review server connection settings.",
    "FTP-Patator": "Block suspicious login attempts and enforce stronger authentication.",
    "Heartbleed": "Check affected systems and update vulnerable OpenSSL versions.",
    "Infiltration": "Isolate the affected device and investigate suspicious network activity.",
    "PortScan": "Review scanned ports and restrict unnecessary network services.",
    "SSH-Patator": "Block suspicious login attempts and enable stronger SSH authentication.",
    WEB_BRUTE_FORCE: "Apply account lockout policies and strengthen authentication.",
    WEB_SQL_INJECTION: "Review application inputs and implement parameterized queries.",
    WEB_XSS: "Validate and sanitize user inputs to prevent script injection."
}


# Generate Detection Results
def generate_result_page(session, page, per_page):

    predictions = session["predictions"]
    confidence_scores = session["confidence_scores"]

    total_records = len(predictions)

    total_pages = max(
        1,
        (total_records + per_page - 1) // per_page
    )

    page = max(1, min(page, total_pages))

    start_index = (page - 1) * per_page
    end_index = min(start_index + per_page, total_records)

    page_predictions = predictions[start_index:end_index]

    page_labels = label_encoder.classes_[page_predictions]

    results = []

    for index, attack_type in enumerate(page_labels):

        attack_type = str(attack_type)
        result_index = start_index + index

        results.append({
            "row": result_index + 1,
            "attack_type": attack_type,
            "risk_level": RISK_LEVELS.get(
                attack_type, "Unknown"
            ),
            "confidence_score": round(
                float(confidence_scores[result_index]), 2
            ),
            "recommendation": RECOMMENDATIONS.get(
                attack_type,
                "Investigate the detected network activity."
            )
        })

    return {
        "results": results,
        "page": page,
        "total_pages": total_pages,
        "total_records": total_records
    }


# Handle Large File Upload Error
@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "status": "error",
        "message": "Uploaded file is too large. Maximum file size is 200 MB."
    }), 413


# Home Route
@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "AI-Based Network Intrusion Detection System API is running."
    })


# Health Route
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "model_loaded": True,
        "number_of_features": len(FEATURES),
        "number_of_classes": len(label_encoder.classes_)
    })


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No CSV file was uploaded."
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "status": "error",
                "message": "No file was selected."
            }), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({
                "status": "error",
                "message": "Please upload a CSV file."
            }), 400

        df = pd.read_csv(file)

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV file is empty."
            }), 400

        # Check Required Features
        missing_features = [
            feature for feature in FEATURES
            if feature not in df.columns
        ]

        if missing_features:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV is missing required model features.",
                "missing_features": missing_features
            }), 400

        # Prepare Model Input
        X = df.loc[:, FEATURES].apply(
            pd.to_numeric,
            errors="coerce"
        ).astype(np.float32)

        if X.isnull().values.any():
            return jsonify({
                "status": "error",
                "message": "The CSV contains missing or non-numeric feature values."
            }), 400

        if not np.isfinite(X.to_numpy()).all():
            return jsonify({
                "status": "error",
                "message": "The CSV contains infinite feature values."
            }), 400

        # Generate Detection Results
        total_records = len(X)

        predictions = np.empty(
            total_records,
            dtype=np.int32
        )

        confidence_scores = np.empty(
            total_records,
            dtype=np.float32
        )

        for start in range(0, total_records, BATCH_SIZE):

            end = min(
                start + BATCH_SIZE,
                total_records
            )

            batch = X.iloc[start:end]

            batch_predictions = model.predict(batch)
            batch_probabilities = model.predict_proba(batch)

            predictions[start:end] = batch_predictions.astype(int)

            confidence_scores[start:end] = (
                np.max(batch_probabilities, axis=1) * 100
            ).astype(np.float32)

        predicted_labels = label_encoder.classes_[predictions]

        # Generate Detection Summary
        summary = dict(
            Counter(str(label) for label in predicted_labels)
        )

        normal_records = summary.get("BENIGN", 0)

        detected_attacks = (
            total_records - normal_records
        )

        # Calculate Ground Truth Metrics
        metrics = None
        ground_truth_available = False

        if "Label" in df.columns:

            y_true = (
                df["Label"]
                .astype(str)
                .str.strip()
            )

            y_pred = pd.Series(
                predicted_labels
            ).astype(str).str.strip()

            valid_labels = set(
                str(label).strip()
                for label in label_encoder.classes_
            )

            unknown_labels = sorted(
                set(y_true.unique()) - valid_labels
            )

            if unknown_labels:

                return jsonify({
                    "status": "error",
                    "message": "The CSV contains ground truth labels that are not supported by the model.",
                    "unknown_labels": unknown_labels
                }), 400

            metrics = {
                "accuracy": round(
                    float(accuracy_score(y_true, y_pred)),
                    4
                ),
                "precision": round(
                    float(precision_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0
                    )),
                    4
                ),
                "recall": round(
                    float(recall_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0
                    )),
                    4
                ),
                "macro_f1": round(
                    float(f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0
                    )),
                    4
                )
            }

            ground_truth_available = True

        # Store Detection Results
        DETECTION_SESSIONS.clear()

        session_id = str(uuid.uuid4())

        DETECTION_SESSIONS[session_id] = {
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "attack_summary": summary,
            "total_records": total_records,
            "normal_records": normal_records,
            "detected_attacks": detected_attacks,
            "metrics": metrics,
            "ground_truth_available": ground_truth_available
        }

        # Return First Page Only
        first_page = generate_result_page(
            DETECTION_SESSIONS[session_id],
            1,
            RECORDS_PER_PAGE
        )

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "total_records": total_records,
            "normal_records": normal_records,
            "detected_attacks": detected_attacks,
            "attack_summary": summary,
            "ground_truth_available": ground_truth_available,
            "metrics": metrics,
            "results": first_page["results"],
            "page": first_page["page"],
            "total_pages": first_page["total_pages"]
        })

    except Exception as error:

        print("Prediction error:", str(error))

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


# Detection Results Pagination Route
@app.route("/results/<session_id>", methods=["GET"])
def get_results(session_id):

    try:

        session = DETECTION_SESSIONS.get(session_id)

        if session is None:
            return jsonify({
                "status": "error",
                "message": "Detection session not found. Please upload the CSV again."
            }), 404

        page = request.args.get(
            "page",
            default=1,
            type=int
        )

        per_page = request.args.get(
            "per_page",
            default=RECORDS_PER_PAGE,
            type=int
        )

        if per_page < 1 or per_page > 100:
            per_page = RECORDS_PER_PAGE

        page_data = generate_result_page(
            session,
            page,
            per_page
        )

        return jsonify({
            "status": "success",
            **page_data
        })

    except Exception as error:

        print("Results pagination error:", str(error))

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


# Excel Report Route
@app.route("/download-report", methods=["POST"])
def download_report():

    temp_path = None

    try:

        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id")

        session = DETECTION_SESSIONS.get(session_id)

        if session is None:
            return jsonify({
                "status": "error",
                "message": "Detection session not found. Please upload the CSV again."
            }), 404

        predictions = session["predictions"]
        confidence_scores = session["confidence_scores"]

        total_records = session["total_records"]
        normal_records = session["normal_records"]
        detected_attacks = session["detected_attacks"]

        attack_summary = session["attack_summary"]

        # Create Excel File
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".xlsx",
            delete=False
        )

        temp_path = temp_file.name
        temp_file.close()

        workbook = xlsxwriter.Workbook(
            temp_path,
            {"constant_memory": True}
        )

        header_format = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter"
        })

        center_format = workbook.add_format({
            "align": "center"
        })

        # Create Detection Summary
        summary_sheet = workbook.add_worksheet(
            "Detection Summary"
        )

        summary_sheet.write_row(
            0,
            0,
            ["Detection Summary", "Records", "Percentage"],
            header_format
        )

        benign_percentage = (
            normal_records / total_records * 100
            if total_records else 0
        )

        attack_percentage = (
            detected_attacks / total_records * 100
            if total_records else 0
        )

        summary_sheet.write_row(
            1,
            0,
            ["Total Records", total_records, 100],
            center_format
        )

        summary_sheet.write_row(
            2,
            0,
            ["BENIGN (Normal Traffic)", normal_records, benign_percentage],
            center_format
        )

        summary_sheet.write_row(
            3,
            0,
            ["Detected Attacks", detected_attacks, attack_percentage],
            center_format
        )

        summary_sheet.write_row(
            5,
            0,
            ["Attack Type", "Records", "Percentage"],
            header_format
        )

        sorted_summary = sorted(
            attack_summary.items(),
            key=lambda item: item[1],
            reverse=True
        )

        for index, (attack_type, count) in enumerate(
            sorted_summary,
            start=6
        ):

            percentage = (
                count / total_records * 100
                if total_records else 0
            )

            summary_sheet.write_row(
                index,
                0,
                [attack_type, count, percentage],
                center_format
            )

        summary_sheet.set_column("A:A", 35)
        summary_sheet.set_column("B:C", 18)

        # Create Detection Chart
        chart = workbook.add_chart({
            "type": "pie"
        })

        chart.add_series({
            "name": "Normal Traffic vs Detected Attacks",
            "categories": ["Detection Summary", 2, 0, 3, 0],
            "values": ["Detection Summary", 2, 1, 3, 1],
            "data_labels": {
                "percentage": True,
                "leader_lines": True
            }
        })

        chart.set_title({
            "name": "Normal Traffic vs Detected Attacks"
        })

        summary_sheet.insert_chart(
            "E2",
            chart
        )

        # Create Detection Results Sheet
        result_sheet = workbook.add_worksheet(
            "Detection Results"
        )

        result_sheet.write_row(
            0,
            0,
            [
                "Row",
                "Attack Type",
                "Risk Level",
                "Confidence Score (%)",
                "Security Recommendation"
            ],
            header_format
        )

        result_sheet.freeze_panes(1, 0)

        result_sheet.set_column("A:A", 12)
        result_sheet.set_column("B:B", 28)
        result_sheet.set_column("C:C", 18)
        result_sheet.set_column("D:D", 24)
        result_sheet.set_column("E:E", 75)

        # Generate All Predicted Labels
        predicted_labels = label_encoder.classes_[predictions]

        # Write Detection Results
        for index, attack_type in enumerate(predicted_labels):

            attack_type = str(attack_type)

            result_sheet.write_row(
                index + 1,
                0,
                [
                    index + 1,
                    attack_type,
                    RISK_LEVELS.get(attack_type, "Unknown"),
                    round(float(confidence_scores[index]), 2),
                    RECOMMENDATIONS.get(
                        attack_type,
                        "Investigate the detected network activity."
                    )
                ]
            )

        result_sheet.autofilter(
            0,
            0,
            total_records,
            4
        )

        workbook.close()

        # Send Excel File
        response = send_file(
            temp_path,
            as_attachment=True,
            download_name="AI_IDS_Detection_Report.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        response.call_on_close(
            lambda: os.path.exists(temp_path) and os.remove(temp_path)
        )

        return response

    except Exception as error:

        print("Excel report error:", str(error))

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


# Start Flask Server
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
