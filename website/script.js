
// Get HTML Elements

const csvFile = document.getElementById("csvFile");
const uploadButton = document.getElementById("uploadButton");
const statusMessage = document.getElementById("statusMessage");

const resultsSection = document.getElementById("resultsSection");
const totalRecords = document.getElementById("totalRecords");
const normalRecords = document.getElementById("normalRecords");
const detectedAttacks = document.getElementById("detectedAttacks");

const modelEvaluationSection = document.getElementById("modelEvaluationSection");
const accuracyMetric = document.getElementById("accuracyMetric");
const precisionMetric = document.getElementById("precisionMetric");
const recallMetric = document.getElementById("recallMetric");
const f1Metric = document.getElementById("f1Metric");

const attackSummary = document.getElementById("attackSummary");
const resultsTable = document.getElementById("resultsTable");
const downloadButton = document.getElementById("downloadButton");

const pageInfo = document.getElementById("pageInfo");
const prevPage = document.getElementById("prevPage");
const nextPage = document.getElementById("nextPage");


// API Settings

const API_URL = "http://127.0.0.1:5000";


// Pagination Settings

const recordsPerPage = 20;

let sessionId = null;
let currentPage = 1;
let totalPages = 1;
let totalResultRecords = 0;


// Handle CSV File Upload

uploadButton.addEventListener("click", async function () {

    const file = csvFile.files[0];

    if (!file) {
        statusMessage.textContent = "Please select a CSV file first.";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        statusMessage.textContent = "Please upload a valid CSV file.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    uploadButton.disabled = true;
    uploadButton.textContent = "Detecting...";
    statusMessage.textContent =
        "Uploading file and analysing network traffic...";

    resultsSection.hidden = true;

    sessionId = null;
    currentPage = 1;
    totalPages = 1;
    totalResultRecords = 0;

    try {

        // Connect to Flask API

        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.status !== "success") {
            throw new Error(
                data.message || data.error || "Detection failed."
            );
        }

        sessionId = data.session_id;
        totalResultRecords = data.total_records;
        totalPages = data.total_pages;

        // Generate Detection Overview

        totalRecords.textContent = data.total_records;
        normalRecords.textContent = data.normal_records;
        detectedAttacks.textContent = data.detected_attacks;

        // Generate Model Evaluation Metrics

        if (data.ground_truth_available && data.metrics) {

            modelEvaluationSection.hidden = false;

            accuracyMetric.textContent =
                (data.metrics.accuracy * 100).toFixed(2) + "%";

            precisionMetric.textContent =
                (data.metrics.precision * 100).toFixed(2) + "%";

            recallMetric.textContent =
                (data.metrics.recall * 100).toFixed(2) + "%";

            f1Metric.textContent =
                (data.metrics.macro_f1 * 100).toFixed(2) + "%";

        } else {

            modelEvaluationSection.hidden = true;
        }

        // Generate Attack Summary

        attackSummary.replaceChildren();

        Object.entries(data.attack_summary).forEach(([attack, count]) => {

            const item = document.createElement("div");
            item.className = "summary-item";

            const name = document.createElement("strong");
            name.textContent = attack;

            const number = document.createElement("span");
            number.textContent = count + " records";

            item.appendChild(name);
            item.appendChild(number);

            attackSummary.appendChild(item);
        });

        // Display First Page

        currentPage = 1;

        await loadDetectionPage(currentPage);

        resultsSection.hidden = false;

        statusMessage.textContent =
            "Detection completed successfully!";

    } catch (error) {

        statusMessage.textContent =
            "Error: " + error.message;

    } finally {

        uploadButton.disabled = false;
        uploadButton.textContent = "Upload and Detect";
    }
});


// Load Detection Results Page

async function loadDetectionPage(page) {

    if (!sessionId) {
        return;
    }

    prevPage.disabled = true;
    nextPage.disabled = true;

    try {

        const response = await fetch(
            `${API_URL}/results/${sessionId}?page=${page}&per_page=${recordsPerPage}`
        );

        const data = await response.json();

        if (!response.ok || data.status !== "success") {
            throw new Error(
                data.message || "Failed to load detection results."
            );
        }

        currentPage = data.page;
        totalPages = data.total_pages;

        renderDetectionResults(data.results);

    } catch (error) {

        statusMessage.textContent =
            "Error loading detection results: " + error.message;

    } finally {

        prevPage.disabled = currentPage <= 1;
        nextPage.disabled = currentPage >= totalPages;
    }
}


// Generate Detection Results

function renderDetectionResults(results) {

    resultsTable.replaceChildren();

    results.forEach(result => {

        const row = document.createElement("tr");

        const values = [
            result.row,
            result.attack_type,
            result.risk_level,
            result.confidence_score + "%",
            result.recommendation
        ];

        values.forEach((value, index) => {

            const cell = document.createElement("td");

            cell.textContent = value;

            // Highlight Attack Type

            if (index === 1) {

                const attackType = String(
                    result.attack_type
                ).toLowerCase();

                if (attackType === "benign") {
                    cell.className = "attack-benign";
                } else {
                    cell.className = "attack-ddos";
                }
            }

            // Highlight Risk Level

            if (index === 2) {

                cell.className =
                    "risk-" +
                    String(result.risk_level).toLowerCase();
            }

            row.appendChild(cell);
        });

        resultsTable.appendChild(row);
    });

    // Update Pagination Information

    if (totalResultRecords === 0) {

        pageInfo.textContent =
            "No detection results available.";

    } else {

        const startIndex =
            (currentPage - 1) * recordsPerPage + 1;

        const endIndex = Math.min(
            currentPage * recordsPerPage,
            totalResultRecords
        );

        pageInfo.textContent =
            `Showing ${startIndex}-${endIndex} of ${totalResultRecords} records | Page ${currentPage} of ${totalPages}`;
    }
}


// Previous Page

prevPage.addEventListener("click", async function () {

    if (currentPage > 1) {

        await loadDetectionPage(currentPage - 1);
    }
});


// Next Page

nextPage.addEventListener("click", async function () {

    if (currentPage < totalPages) {

        await loadDetectionPage(currentPage + 1);
    }
});


// Download Excel Detection Report

downloadButton.addEventListener("click", async function () {

    if (!sessionId) {

        statusMessage.textContent =
            "No detection results available to download.";

        return;
    }

    downloadButton.disabled = true;
    downloadButton.textContent = "Generating Excel Report...";

    try {

        // Send Session ID to Flask API

        const response = await fetch(
            `${API_URL}/download-report`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    session_id: sessionId
                })
            }
        );

        if (!response.ok) {

            let errorMessage =
                "Failed to generate Excel report.";

            try {

                const errorData = await response.json();

                errorMessage =
                    errorData.message ||
                    errorData.error ||
                    errorMessage;

            } catch (error) {
                // Keep the default error message
            }

            throw new Error(errorMessage);
        }

        // Receive Excel File

        const blob = await response.blob();

        // Download Excel Report

        const url = URL.createObjectURL(blob);

        const link = document.createElement("a");

        link.href = url;
        link.download = "AI_IDS_Detection_Report.xlsx";

        document.body.appendChild(link);

        link.click();

        document.body.removeChild(link);

        setTimeout(() => {
            URL.revokeObjectURL(url);
        }, 1000);

        statusMessage.textContent =
            "Excel detection report downloaded successfully!";

    } catch (error) {

        statusMessage.textContent =
            "Error: " + error.message;

    } finally {

        downloadButton.disabled = false;

        downloadButton.textContent =
            "Download Detection Results";
    }
});
