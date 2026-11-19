# Flask API

import json
import joblib
import numpy as np
import pandas as pd

from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import PieChart, Reference
from openpyxl.chart.label import DataLabelList


# Initialize Flask Application

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# Define Project Paths

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "xgboost_model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "label_encoder.pkl"
MAPPING_PATH = BASE_DIR / "model" / "label_mapping.json"
FEATURE_PATH = BASE_DIR / "training" / "selected_feature.json"


# Load Trained Model and Supporting Files

print("Loading AI-Based Network Intrusion Detection System...")

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)

with open(MAPPING_PATH, "r", encoding="utf-8") as file:
    label_mapping = json.load(file)

with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    feature_data = json.load(file)


# Get the Features Used by the Trained Model

SELECTED_FEATURES = feature_data["features"]

MODEL_FEATURES = model.get_booster().feature_names

if MODEL_FEATURES is None:
    raise ValueError("Model feature names are not available.")

if set(SELECTED_FEATURES) != set(MODEL_FEATURES):
    raise ValueError("Selected features do not match model features.")

FEATURES = MODEL_FEATURES

print("Model loaded successfully!")
print("Number of features:", len(FEATURES))
print("Number of attack classes:", len(label_mapping))


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
    "Web Attack - Brute Force": "High",
    "Web Attack - Sql Injection": "Critical",
    "Web Attack - XSS": "High"
}


# Define Security Recommendations

RECOMMENDATIONS = {
    "BENIGN": "No suspicious activity detected. Continue monitoring network traffic.",

    "Bot": "Investigate the affected device, isolate it if necessary, and scan for malware.",

    "DDoS": "Apply rate limiting, activate DDoS protection, and investigate abnormal traffic sources.",

    "DoS GoldenEye": "Investigate the source IP and apply firewall rules to block malicious traffic.",

    "DoS Hulk": "Apply traffic filtering and rate limiting to protect the targeted server.",

    "DoS Slowhttptest": "Review incomplete HTTP connections and configure server connection timeouts.",

    "DoS slowloris": "Limit concurrent connections and configure appropriate server timeouts.",

    "FTP-Patator": "Review FTP login attempts, block suspicious IP addresses, and enforce strong passwords.",

    "Heartbleed": "Check affected systems, patch vulnerable OpenSSL versions, and review exposed credentials.",

    "Infiltration": "Investigate suspicious connections, isolate affected systems, and review security logs.",

    "PortScan": "Monitor repeated connection attempts and restrict unnecessary open ports.",

    "SSH-Patator": "Block suspicious login sources and enforce strong authentication or multi-factor authentication.",

    "Web Attack - Brute Force": "Implement login rate limiting, account lockout policies, and multi-factor authentication.",

    "Web Attack - Sql Injection": "Review application inputs and use parameterized SQL queries and a web application firewall.",

    "Web Attack - XSS": "Validate and sanitize user inputs and apply appropriate output encoding."
}


# Define Home Route

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "AI-Based Network Intrusion Detection System API",
        "status": "Running",
        "model": "XGBoost",
        "features": len(FEATURES),
        "classes": len(label_mapping)
    })


# Define Health Check Route

@app.route("/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# Handle CSV Upload and Intrusion Detection

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No CSV file uploaded."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    try:

        # Read Uploaded CSV File

        df = pd.read_csv(file)

        if df.empty:
            return jsonify({
                "error": "The uploaded CSV file is empty."
            }), 400

        # Validate Required Features

        missing_features = [
            feature for feature in FEATURES
            if feature not in df.columns
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features in CSV file.",
                "missing_features": missing_features
            }), 400

        # Select Required Features

        X = df[FEATURES].copy()

        # Convert Features into Numeric Values

        for column in FEATURES:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        # Check Invalid Feature Values

        invalid_mask = ~np.isfinite(
            X.to_numpy(dtype=float)
        )

        if invalid_mask.any():

            invalid_rows = np.where(
                invalid_mask.any(axis=1)
            )[0]

            return jsonify({
                "error": "CSV contains missing or invalid feature values.",
                "invalid_row_count": int(len(invalid_rows)),
                "example_row_numbers": (
                    invalid_rows[:10] + 1
                ).tolist()
            }), 400

        # Perform XGBoost Prediction

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)

        # Convert Predictions into Attack Labels

        predicted_labels = label_encoder.inverse_transform(
            predictions.astype(int)
        )

        # Calculate Confidence Scores

        confidence_scores = np.max(
            probabilities,
            axis=1
        ) * 100

        # Generate Detection Results

        results = []

        for index, attack_type in enumerate(predicted_labels):

            attack_type = str(attack_type)

            risk_level = RISK_LEVELS.get(
                attack_type,
                "Unknown"
            )

            recommendation = RECOMMENDATIONS.get(
                attack_type,
                "Investigate the detected network activity."
            )

            results.append({
                "row": index + 1,
                "attack_type": attack_type,
                "risk_level": risk_level,
                "confidence_score": round(
                    float(confidence_scores[index]), 2
                ),
                "recommendation": recommendation
            })

        # Generate Detection Summary

        summary = {}

        for result in results:

            attack = result["attack_type"]

            summary[attack] = summary.get(attack, 0) + 1

        detected_attacks = sum(
            count for attack, count in summary.items()
            if attack != "BENIGN"
        )

        # Return Detection Results

        return jsonify({
            "status": "success",
            "total_records": len(results),
            "normal_records": summary.get("BENIGN", 0),
            "detected_attacks": detected_attacks,
            "attack_summary": summary,
            "results": results
        })

    except Exception as error:

        print("Prediction error:", str(error))

        return jsonify({
            "error": "An error occurred during prediction.",
            "details": str(error)
        }), 500


# Generate Excel Detection Report

@app.route("/download-report", methods=["POST"])
def download_report():

    try:

        data = request.get_json()

        if not data or not isinstance(data.get("results"), list):
            return jsonify({
                "error": "No detection results provided."
            }), 400

        results = data["results"]

        if not results:
            return jsonify({
                "error": "Detection results are empty."
            }), 400

        total_records = len(results)

        normal_records = sum(
            1 for result in results
            if result.get("attack_type") == "BENIGN"
        )

        detected_attacks = total_records - normal_records

        benign_percentage = normal_records / total_records
        attack_percentage = detected_attacks / total_records

        # Create Excel Workbook

        workbook = Workbook()

        summary_sheet = workbook.active
        summary_sheet.title = "Detection Summary"

        results_sheet = workbook.create_sheet(
            "Detection Results"
        )

        # Define Excel Styles

        header_fill = PatternFill(
            "solid",
            fgColor="1F4E78"
        )

        header_font = Font(
            color="FFFFFF",
            bold=True
        )

        title_font = Font(
            size=16,
            bold=True,
            color="1F4E78"
        )

        green_fill = PatternFill(
            "solid",
            fgColor="C6EFCE"
        )

        red_fill = PatternFill(
            "solid",
            fgColor="FFC7CE"
        )

        # Detection Summary Sheet

        summary_sheet["A1"] = "AI-Based Network Intrusion Detection System"
        summary_sheet["A1"].font = title_font

        summary_sheet["A2"] = "Detection Results Summary"
        summary_sheet["A2"].font = Font(
            size=12,
            bold=True
        )

        summary_sheet.append([])

        summary_sheet.append([
            "Detection Category",
            "Records",
            "Percentage"
        ])

        summary_data = [
            ["Total Records", total_records, 1],
            ["BENIGN (Normal Traffic)", normal_records, benign_percentage],
            ["DDoS (Detected Attack)", detected_attacks, attack_percentage]
        ]

        for row in summary_data:
            summary_sheet.append(row)

        # Format Summary Table

        for cell in summary_sheet[4]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center"
            )

        for row in range(5, 8):
            summary_sheet.cell(
                row,
                3
            ).number_format = "0.0%"

        for column in range(1, 4):
            summary_sheet.cell(6, column).fill = green_fill
            summary_sheet.cell(7, column).fill = red_fill

        for row in range(5, 8):
            for column in range(1, 4):
                summary_sheet.cell(
                    row,
                    column
                ).alignment = Alignment(
                    vertical="center"
                )

        summary_sheet.column_dimensions["A"].width = 35
        summary_sheet.column_dimensions["B"].width = 15
        summary_sheet.column_dimensions["C"].width = 18

        # Create Pie Chart

        pie_chart = PieChart()

        pie_chart.title = "Normal Traffic vs Detected Attacks"
        pie_chart.style = 10
        pie_chart.height = 9
        pie_chart.width = 15

        pie_data = Reference(
            summary_sheet,
            min_col=2,
            min_row=6,
            max_row=7
        )

        pie_labels = Reference(
            summary_sheet,
            min_col=1,
            min_row=6,
            max_row=7
        )

        pie_chart.add_data(
            pie_data,
            titles_from_data=False
        )

        pie_chart.set_categories(pie_labels)

        pie_chart.dataLabels = DataLabelList()
        pie_chart.dataLabels.showPercent = True
        pie_chart.dataLabels.showVal = True

        summary_sheet.add_chart(
            pie_chart,
            "E4"
        )

        # Detection Results Sheet

        headers = [
            "Row",
            "Attack Type",
            "Risk Level",
            "Confidence Score",
            "Security Recommendation"
        ]

        results_sheet.append(headers)

        for cell in results_sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )

        for result in results:

            results_sheet.append([
                result.get("row", ""),
                result.get("attack_type", ""),
                result.get("risk_level", ""),
                float(result.get("confidence_score", 0)) / 100,
                result.get("recommendation", "")
            ])

            current_row = results_sheet.max_row

            results_sheet.cell(
                current_row,
                4
            ).number_format = "0.00%"

            attack_type = result.get("attack_type", "")

            if attack_type == "BENIGN":

                fill = green_fill

            else:

                fill = red_fill

            for column in [2, 3]:
                results_sheet.cell(
                    current_row,
                    column
                ).fill = fill

        # Format Detection Results

        results_sheet.column_dimensions["A"].width = 10
        results_sheet.column_dimensions["B"].width = 25
        results_sheet.column_dimensions["C"].width = 18
        results_sheet.column_dimensions["D"].width = 22
        results_sheet.column_dimensions["E"].width = 85

        results_sheet.freeze_panes = "A2"
        results_sheet.auto_filter.ref = results_sheet.dimensions

        # Save Excel File in Memory

        output = BytesIO()

        workbook.save(output)
        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="AI_IDS_Detection_Report.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as error:

        print("Excel report error:", str(error))

        return jsonify({
            "error": "Failed to generate Excel report.",
            "details": str(error)
        }), 500


# Run Flask Application

if __name__ == "__main__":

    print("AI-Based Network Intrusion Detection System")
    print("Flask API Server")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
