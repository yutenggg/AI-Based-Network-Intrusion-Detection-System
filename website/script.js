
// Get HTML Elements

const csvFile = document.getElementById("csvFile");
const uploadButton = document.getElementById("uploadButton");
const statusMessage = document.getElementById("statusMessage");

const resultsSection = document.getElementById("resultsSection");
const totalRecords = document.getElementById("totalRecords");
const normalRecords = document.getElementById("normalRecords");
const detectedAttacks = document.getElementById("detectedAttacks");

const attackSummary = document.getElementById("attackSummary");
const resultsTable = document.getElementById("resultsTable");
const downloadButton = document.getElementById("downloadButton");

const pageInfo = document.getElementById("pageInfo");
const prevPage = document.getElementById("prevPage");
const nextPage = document.getElementById("nextPage");


// Pagination Settings

const recordsPerPage = 20;

let detectionResults = [];
let currentPage = 1;


// Handle CSV File Upload

uploadButton.addEventListener("click", async function () {

    const file = csvFile.files[0];

    // Validate CSV File

    if (!file) {
        statusMessage.textContent = "Please select a CSV file first.";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        statusMessage.textContent = "Please upload a valid CSV file.";
        return;
    }

    // Prepare File Data

    const formData = new FormData();
    formData.append("file", file);

    uploadButton.disabled = true;
    uploadButton.textContent = "Detecting...";
    statusMessage.textContent = "Uploading file and analysing network traffic...";

    resultsSection.hidden = true;

    detectionResults = [];
    currentPage = 1;

    try {

        // Connect to Flask API

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.status !== "success") {
            throw new Error(data.error || "Detection failed.");
        }

        // Generate Detection Overview

        totalRecords.textContent = data.total_records;
        normalRecords.textContent = data.normal_records;
        detectedAttacks.textContent = data.detected_attacks;

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

        // Store Detection Results

        detectionResults = data.results;
        currentPage = 1;

        // Display Detection Results

        renderDetectionResults();

        resultsSection.hidden = false;
        statusMessage.textContent = "Detection completed successfully!";

    } catch (error) {

        // Handle Detection Errors

        statusMessage.textContent = "Error: " + error.message;

    } finally {

        // Reset Upload Button

        uploadButton.disabled = false;
        uploadButton.textContent = "Upload and Detect";
    }
});


// Generate Detection Results

function renderDetectionResults() {

    resultsTable.replaceChildren();

    const totalPages = Math.ceil(
        detectionResults.length / recordsPerPage
    );

    const startIndex = (currentPage - 1) * recordsPerPage;

    const endIndex = Math.min(
        startIndex + recordsPerPage,
        detectionResults.length
    );

    const currentResults = detectionResults.slice(
        startIndex,
        endIndex
    );

    currentResults.forEach(result => {

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

    if (detectionResults.length === 0) {

        pageInfo.textContent = "No detection results available.";

    } else {

        pageInfo.textContent =
            `Showing ${startIndex + 1}-${endIndex} of ${detectionResults.length} records | Page ${currentPage} of ${totalPages}`;
    }

    // Update Pagination Buttons

    prevPage.disabled = currentPage <= 1;
    nextPage.disabled = currentPage >= totalPages;
}


// Previous Page

prevPage.addEventListener("click", function () {

    if (currentPage > 1) {

        currentPage--;

        renderDetectionResults();
    }
});


// Next Page

nextPage.addEventListener("click", function () {

    const totalPages = Math.ceil(
        detectionResults.length / recordsPerPage
    );

    if (currentPage < totalPages) {

        currentPage++;

        renderDetectionResults();
    }
});


// Download Excel Detection Report

downloadButton.addEventListener("click", async function () {

    if (detectionResults.length === 0) {

        statusMessage.textContent =
            "No detection results available to download.";

        return;
    }

    downloadButton.disabled = true;
    downloadButton.textContent = "Generating Excel Report...";

    try {

        // Send Detection Results to Flask API

        const response = await fetch(
            "http://127.0.0.1:5000/download-report",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    results: detectionResults
                })
            }
        );

        if (!response.ok) {

            const errorData = await response.json();

            throw new Error(
                errorData.error ||
                "Failed to generate Excel report."
            );
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
