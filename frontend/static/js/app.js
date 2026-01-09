// ================================
// SOCKET.IO CONNECTION
// ================================
const socket = io({
    transports: ['polling'],  // Use polling only, disable WebSocket
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    reconnectionAttempts: 5
});

// ================================
// DOM ELEMENTS
// ================================
const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("start-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const downloadBtn = document.getElementById("download-btn");

const bufferText = document.getElementById("buffer-text");
const bufferProgress = document.getElementById("buffer-progress");

const riskPercentage = document.getElementById("risk-percentage");
const riskLevel = document.getElementById("risk-level");

const analysisCountEl = document.getElementById("analysis-count");
const avgRiskEl = document.getElementById("avg-risk");
const lastUpdateEl = document.getElementById("last-update");

const auList = document.getElementById("au-list");
const explanationText = document.getElementById("explanation-text");

const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = loadingOverlay.querySelector("p");

// ================================
// STATE
// ================================
let stream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let analysisCount = 0;
let riskHistory = [];
let clientId = null;

// ================================
// CAMERA
// ================================
startBtn.addEventListener("click", async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcam.srcObject = stream;
        analyzeBtn.disabled = false;
        startBtn.textContent = "📹 Camera Started";
        startBtn.disabled = true;
    } catch (error) {
        alert("Camera access denied: " + error.message);
    }
});

// ================================
// START RECORDING
// ================================
analyzeBtn.addEventListener("click", () => {
    if (isRecording) {
        stopRecordingAndAnalyze();
    } else {
        startRecording();
    }
});

function startRecording() {
    recordedChunks = [];
    
    // Create MediaRecorder to capture video stream
    const options = { mimeType: "video/webm;codecs=vp9" };
    
    // Fallback if vp9 not supported
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = "video/webm;codecs=vp8";
    }
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = "video/webm";
    }

    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        console.log("Recording stopped");
    };

    mediaRecorder.start();
    isRecording = true;
    analyzeBtn.textContent = "⏹️ Stop & Analyze";
    bufferText.textContent = "🔴 Recording...";
    bufferProgress.style.width = "100%";
}

function stopRecordingAndAnalyze() {
    if (!mediaRecorder) return;

    mediaRecorder.stop();
    isRecording = false;
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "🔍 Analyze Now";

    // Show loading overlay
    loadingOverlay.style.display = "flex";
    loadingText.textContent = "Preparing video for upload...";

    // Wait a moment for ondataavailable to complete
    setTimeout(() => {
        sendVideoToBackend();
    }, 500);
}

// ================================
// SEND VIDEO TO BACKEND
// ================================
async function sendVideoToBackend() {
    try {
        loadingText.textContent = "Uploading video...";

        // Create blob from recorded chunks
        const blob = new Blob(recordedChunks, { type: "video/webm" });
        console.log(`Video blob size: ${blob.size} bytes`);

        // Split blob into chunks (1MB each)
        const CHUNK_SIZE = 1024 * 1024; // 1MB
        const chunks = [];

        for (let i = 0; i < blob.size; i += CHUNK_SIZE) {
            chunks.push(blob.slice(i, i + CHUNK_SIZE));
        }

        console.log(`Sending ${chunks.length} chunks`);

        // Send each chunk to backend
        for (let i = 0; i < chunks.length; i++) {
            const chunkData = await chunks[i].arrayBuffer();
            const hexData = Array.from(new Uint8Array(chunkData))
                .map(b => b.toString(16).padStart(2, "0"))
                .join("");

            socket.emit("video_chunk", {
                chunk_id: i,
                chunk_data: hexData,
                is_last: i === chunks.length - 1
            });

            // Update progress
            const progress = Math.round(((i + 1) / chunks.length) * 100);
            bufferText.textContent = `Upload: ${progress}%`;
            bufferProgress.style.width = `${progress}%`;
        }

        console.log("All chunks sent, waiting for backend to process...");
        loadingText.textContent = "Uploading complete. Processing video...";

        // Wait a moment for all events to be received
        setTimeout(() => {
            // Send analyze command with total chunk count
            console.log(`📤 Sending analyze event with ${chunks.length} chunks`);
            socket.emit("analyze", {
                total_chunks: chunks.length
            }, (ack) => {
                console.log("Analyze event acknowledged:", ack);
            });
        }, 500);

    } catch (error) {
        console.error("Error sending video:", error);
        loadingOverlay.style.display = "none";
        alert("Error uploading video: " + error.message);
        analyzeBtn.disabled = false;
    }
}

// ================================
// SOCKET EVENTS - CONNECTION
// ================================
socket.on("connect", () => {
    console.log("✅ Connected to server");
});

socket.on("disconnect", () => {
    console.log("⚠️ Disconnected from server");
});

socket.on("connect_error", (error) => {
    console.error("❌ Connection error:", error);
});

socket.on("response", data => {
    console.log("Server response:", data);
});

socket.on("chunks_received", data => {
    console.log("Backend ready to process chunks");
    loadingText.textContent = "Processing video...";
});

socket.on("processing_started", data => {
    loadingOverlay.style.display = "flex";
    loadingText.textContent = "Processing started...";
    console.log(data.status);
});

socket.on("processing_update", data => {
    loadingText.textContent = data.status;
    console.log(data.status);
});

// ================================
// ANALYSIS RESULT
// ================================
socket.on("analysis_complete", data => {
    console.log("Analysis complete:", data);

    if (data.status !== "success") {
        alert("Analysis failed: " + data.message);
        loadingOverlay.style.display = "none";
        analyzeBtn.disabled = false;
        return;
    }

    // Use the probability directly from backend (already 0-1 scale)
    const probability = data.probability;
    const confidencePercent = Math.round(probability * 100);

    analysisCount++;
    riskHistory.push(confidencePercent);

    // Risk UI
    riskPercentage.textContent = `${confidencePercent}%`;
    riskLevel.textContent = getRiskLevel(confidencePercent);

    // Stats
    analysisCountEl.textContent = analysisCount;
    avgRiskEl.textContent =
        Math.round(riskHistory.reduce((a, b) => a + b, 0) / riskHistory.length) + "%";
    lastUpdateEl.textContent = new Date().toLocaleTimeString();

    // Extract and display action units (AUs) from the top_action_units
    auList.innerHTML = "";
    if (data.feature_stats && data.feature_stats.top_action_units) {
        const topAUs = data.feature_stats.top_action_units;
        if (topAUs.length > 0) {
            topAUs.forEach(au => {
                const chip = document.createElement("div");
                chip.className = "au-chip";
                chip.textContent = `${au.name} (${au.intensity.toFixed(2)})`;
                auList.appendChild(chip);
            });
        } else {
            auList.innerHTML = "<p class='placeholder-text'>No action units detected</p>";
        }
    } else {
        auList.innerHTML = "<p class='placeholder-text'>No AU data available</p>";
    }

    // Display explanation from backend
    explanationText.textContent = data.explanation;

    // Update chart
    updateChart(confidencePercent);

    // Hide loading overlay
    loadingOverlay.style.display = "none";
    analyzeBtn.disabled = false;
    downloadBtn.disabled = false;  // Enable download button

    // Reset for next recording
    recordedChunks = [];
    bufferText.textContent = "Ready for next analysis";
    bufferProgress.style.width = "0%";
});

// ================================
// ERROR HANDLING
// ================================
socket.on("error", data => {
    console.error("Error from backend:", data);
    loadingOverlay.style.display = "none";
    analyzeBtn.disabled = false;
    
    let errorMsg = data.message || "Unknown error";
    if (data.error_type === "timeout") {
        errorMsg += "\n\nTry with a shorter video (under 30 seconds).";
    }
    alert("Error: " + errorMsg);
});

// ================================
// UTILITY FUNCTIONS
// ================================
function getRiskLevel(percentage) {
    if (percentage >= 70) return "🔴 High Risk";
    if (percentage >= 50) return "🟡 Moderate Risk";
    if (percentage >= 30) return "🟠 Low Risk";
    return "🟢 Minimal Risk";
}

// ================================
// DOWNLOAD/SAVE RESULTS
// ================================
downloadBtn.addEventListener("click", () => {
    if (analysisCount === 0) {
        alert("No analyses to save yet");
        return;
    }
    
    // Create summary CSV from prediction history
    let csvContent = "timestamp,probability,risk_level,explanation,total_frames,frames_analyzed\n";
    
    // Get data from riskHistory and analysisCount
    const now = new Date();
    const timestamp = now.toISOString();
    const avgRisk = Math.round(riskHistory.reduce((a, b) => a + b, 0) / riskHistory.length);
    
    // Add current analysis
    csvContent += `${timestamp},${(avgRisk / 100).toFixed(4)},LOW,"Analysis completed with ${analysisCount} session(s)",${analysisCount},${analysisCount}\n`;
    
    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `depression_analysis_${now.getTime()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    console.log(`📥 Results saved to: ${link.download}`);
    alert(`Results saved: ${link.download}`);
});

// ================================
// CHART.JS
// ================================
const historyCtx = document.getElementById("history-chart").getContext("2d");

const historyChart = new Chart(historyCtx, {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Confidence Score (%)",
            data: [],
            borderColor: "#6366f1",
            backgroundColor: "rgba(99, 102, 241, 0.1)",
            tension: 0.3,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
            y: {
                min: 0,
                max: 100,
                title: {
                    display: true,
                    text: "Confidence (%)"
                }
            }
        },
        plugins: {
            legend: {
                display: true,
                labels: {
                    color: "#e5e7eb"
                }
            }
        }
    }
});

function updateChart(value) {
    historyChart.data.labels.push(`Analysis ${analysisCount}`);
    historyChart.data.datasets[0].data.push(value);
    historyChart.update();
}