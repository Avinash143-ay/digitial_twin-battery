const csvInput = document.getElementById("csvInput");
const currentInput = document.getElementById("currentInput");
const constantCurrentInput = document.getElementById("constantCurrent");
const currentStepsInput = document.getElementById("currentSteps");
const generateCurrentBtn = document.getElementById("generateCurrentBtn");
const statusBar = document.getElementById("statusBar");
const statusText = document.getElementById("statusText");
const predictBtn = document.getElementById("predictBtn");
const currentInfo = document.getElementById("currentInfo");

// Parameter inputs
const sohInput = document.getElementById("soh");
const voltageInput = document.getElementById("voltage");
const temperatureInput = document.getElementById("temperature");

const requiredHeaders = [
  "voltage_actual",
  "voltage_median_pred",
  "temperature_actual",
  "temperature_median_pred",
];

const isLocalHost =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
const isFileProtocol = window.location.protocol === "file:";

// When opened directly as file://, API calls need an explicit backend base URL.
const API_BASE = isLocalHost || isFileProtocol ? "http://localhost:5000" : "";

const apiUrl = (path) => `${API_BASE}${path}`;

const quantizationModeText = document.getElementById("quantizationModeText");
const quantizationEnabledText = document.getElementById("quantizationEnabledText");
const quantizationHint = document.getElementById("quantizationHint");
const quantizationEnableBtn = document.getElementById("quantizationEnableBtn");
const quantizationDisableBtn = document.getElementById("quantizationDisableBtn");
const quantizationRefreshBtn = document.getElementById("quantizationRefreshBtn");

const setQuantizationUiLoading = (loading) => {
  quantizationEnableBtn.disabled = loading;
  quantizationDisableBtn.disabled = loading;
  quantizationRefreshBtn.disabled = loading;
};

const renderQuantizationInfo = (info) => {
  const mode = info?.ensemble_inference_mode || "unknown";
  const enabled = Boolean(info?.ensemble_quantization_enabled);

  quantizationModeText.textContent = mode === "int8_dynamic" ? "INT8 Dynamic" : "FP32";
  quantizationEnabledText.textContent = enabled ? "Yes" : "No";
  quantizationHint.textContent = enabled
    ? "INT8 mode is active for ensemble inference."
    : "FP32 mode is active for ensemble inference.";

  quantizationEnableBtn.disabled = enabled;
  quantizationDisableBtn.disabled = !enabled;
};

const fetchQuantizationInfo = async () => {
  try {
    const response = await fetch(apiUrl('/quantization_info'));
    if (!response.ok) {
      throw new Error('Unable to fetch quantization status');
    }
    const result = await response.json();
    if (result.status !== 'success') {
      throw new Error(result.message || 'Quantization status unavailable');
    }
    renderQuantizationInfo(result);
  } catch (error) {
    quantizationModeText.textContent = "Unavailable";
    quantizationEnabledText.textContent = "Unavailable";
    quantizationHint.textContent = `Could not reach backend: ${error.message}`;
  }
};

const setQuantizationEnabled = async (enabled) => {
  setQuantizationUiLoading(true);
  try {
    const response = await fetch(apiUrl('/quantization_config'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled })
    });

    const result = await response.json();
    if (!response.ok || result.status !== 'success') {
      throw new Error(result.message || 'Failed to update quantization mode');
    }

    renderQuantizationInfo(result);
    updateStatus(
      enabled ? 'INT8 quantization enabled for ensemble inference' : 'FP32 mode enabled for ensemble inference',
      'loaded'
    );
  } catch (error) {
    quantizationHint.textContent = `Update failed: ${error.message}`;
    updateStatus(`Quantization update failed: ${error.message}`, 'ready');
  } finally {
    setQuantizationUiLoading(false);
  }
};

quantizationEnableBtn.addEventListener('click', () => setQuantizationEnabled(true));
quantizationDisableBtn.addEventListener('click', () => setQuantizationEnabled(false));
quantizationRefreshBtn.addEventListener('click', fetchQuantizationInfo);

fetchQuantizationInfo();

// Separate chart instances for each tab
let predictVoltageChart;
let predictTemperatureChart;
let compareVoltageChart;
let compareTemperatureChart;
let ensembleVoltageChart;
let ensembleTemperatureChart;

let currentData = null;

// Current mode switching
document.querySelectorAll('input[name="currentMode"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    const mode = e.target.value;
    document.getElementById('constantCurrentSection').style.display = mode === 'constant' ? 'block' : 'none';
    document.getElementById('csvCurrentSection').style.display = mode === 'csv' ? 'block' : 'none';
    
    // Reset current data
    currentData = null;
    currentInfo.classList.remove('show');
    predictBtn.disabled = mode === 'csv';
  });
});

// Generate constant current profile
generateCurrentBtn.addEventListener('click', () => {
  const current = parseFloat(constantCurrentInput.value);
  const steps = parseInt(currentStepsInput.value);
  
  currentData = new Array(steps).fill(current);
  
  currentInfo.innerHTML = `✓ Generated ${steps} constant current values (${current}A for ${steps}s)`;
  currentInfo.classList.add('show');
  predictBtn.disabled = false;
  updateStatus(`Current profile ready - ${steps} steps at ${current}A`, 'loaded');
});

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tabName = btn.dataset.tab;
    
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(tabName + 'Tab').classList.add('active');
    
    // Update status
    if (tabName === 'predict') {
      updateStatus('Ready - Upload current data and set initial conditions', 'ready');
    } else if (tabName === 'datasetCompare') {
      updateStatus('Ready - Load a segment from KIT dataset to compare models', 'ready');
    } else if (tabName === 'compare') {
      updateStatus('Ready - Upload CSV for comparison', 'ready');
    } else if (tabName === 'ensemble') {
      updateStatus('Ready - Set parameters and generate current profile', 'ready');
    }
  });
});

// Status update function
const updateStatus = (message, type = 'ready') => {
  statusText.textContent = message;
  statusBar.className = `status-bar ${type}`;
};

// Handle current CSV upload
currentInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    currentData = null;
    predictBtn.disabled = true;
    currentInfo.classList.remove('show');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const text = String(e.target.result || "");
    const { headers, rows } = parseCsv(text);
    
    if (!headers.includes('current')) {
      updateStatus('Error: CSV must contain a "current" column', 'ready');
      currentData = null;
      predictBtn.disabled = true;
      currentInfo.classList.remove('show');
      return;
    }
    
    const currentIndex = headers.indexOf('current');
    currentData = rows.map(row => parseFloat(row[currentIndex])).filter(val => !isNaN(val));
    
    currentInfo.innerHTML = `✓ Loaded ${currentData.length} current values (${currentData.length}s duration)`;
    currentInfo.classList.add('show');
    predictBtn.disabled = false;
    updateStatus(`Current data loaded - ${currentData.length} samples ready`, 'loaded');
  };
  reader.readAsText(file);
});

// Prediction function (connects to Python backend)
const predictBatteryBehavior = async () => {
  if (!currentData || currentData.length === 0) {
    updateStatus('Error: Please upload current data first', 'ready');
    return;
  }

  const soh = parseFloat(sohInput.value);
  const voltage = parseFloat(voltageInput.value);
  const temperature = parseFloat(temperatureInput.value);
  const steps = currentData.length;

  updateStatus(`Generating forecast for ${steps} steps...`, "live");
  predictBtn.disabled = true;

  try {
    // Call Python backend API
    const response = await fetch(apiUrl('/predict'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        soh,
        voltage,
        temperature,
        current_data: currentData,
        steps
      })
    });

    if (!response.ok) {
      throw new Error('Prediction failed');
    }

    const result = await response.json();

    if (result.status === 'success') {
      const voltageForecast = result.voltage_forecast;
      const temperatureForecast = result.temperature_forecast;

      // Create time labels
      const labels = Array.from({ length: steps }, (_, idx) => `${idx}s`);

      destroyPredictCharts();

      const voltageCtx = document.getElementById("predictVoltageChart");
      const temperatureCtx = document.getElementById("predictTemperatureChart");

      // Show forecast as predicted line
      predictVoltageChart = createChart(voltageCtx, labels, [], voltageForecast, "Voltage");
      predictTemperatureChart = createChart(temperatureCtx, labels, [], temperatureForecast, "Temperature");

      document.getElementById("predictVoltageSummary").textContent = `Forecast: ${steps} steps (${steps}s)`;
      document.getElementById("predictTemperatureSummary").textContent = `Forecast: ${steps} steps (${steps}s)`;

      document.getElementById("predictDownloadVoltage").disabled = false;
      document.getElementById("predictDownloadTemperature").disabled = false;

      updateStatus(`Forecast Complete - ${steps} steps (${steps}s) | SOH: ${soh}`, "complete");
    } else {
      throw new Error(result.message || 'Unknown error');
    }

  } catch (error) {
    console.error('Prediction error:', error);
    updateStatus(`Error: ${error.message}. Make sure backend server is running on port 5000.`, "ready");
  } finally {
    predictBtn.disabled = currentData === null;
  }
};

predictBtn.addEventListener("click", predictBatteryBehavior);

const parseCsvLine = (line, delimiter) => {
  const cells = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];

    if (ch === '"') {
      // Handle escaped quotes "" inside quoted fields.
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === delimiter && !inQuotes) {
      cells.push(current.trim());
      current = "";
      continue;
    }

    current += ch;
  }

  cells.push(current.trim());
  return cells;
};

const parseCsv = (text) => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    return { headers: [], rows: [] };
  }

  const headerLine = lines[0];
  const commaCount = (headerLine.match(/,/g) || []).length;
  const semicolonCount = (headerLine.match(/;/g) || []).length;
  const delimiter = semicolonCount > commaCount ? ';' : ',';

  const headers = parseCsvLine(headerLine, delimiter).map((h) => h.trim().replace(/^"|"$/g, ""));
  const rows = lines
    .slice(1)
    .map((line) => parseCsvLine(line, delimiter).map((cell) => cell.trim().replace(/^"|"$/g, "")));

  return { headers, rows };
};

const getColumnIndexMap = (headers) => {
  const map = {};
  headers.forEach((header, index) => {
    map[header] = index;
  });
  return map;
};

const toNumber = (value) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const buildSeries = (rows, indexMap, columnName) =>
  rows
    .map((row) => toNumber(row[indexMap[columnName]]))
    .filter((value) => value !== null);

// Helper function to find voltage threshold crossings
const findVoltageThresholds = (voltageData, labels, title) => {
  const annotations = {};
  
  if (!title.toLowerCase().includes('voltage')) {
    return annotations;
  }
  
  // Find first occurrence of 4.2V threshold
  const maxIndex = voltageData.findIndex(v => v >= 4.2);
  if (maxIndex !== -1) {
    annotations.maxVoltage = {
      type: 'line',
      xMin: maxIndex,
      xMax: maxIndex,
      borderColor: 'rgba(220, 38, 38, 0.8)',
      borderWidth: 2,
      borderDash: [6, 6],
      label: {
        display: true,
        content: `4.2V at ${labels[maxIndex] || maxIndex}s`,
        position: 'start',
        backgroundColor: 'rgba(220, 38, 38, 0.8)',
        color: 'white',
        font: {
          size: 11,
          weight: 'bold'
        },
        padding: 6
      }
    };
  }
  
  // Find first occurrence of 2.5V threshold
  const minIndex = voltageData.findIndex(v => v <= 2.5);
  if (minIndex !== -1) {
    annotations.minVoltage = {
      type: 'line',
      xMin: minIndex,
      xMax: minIndex,
      borderColor: 'rgba(220, 38, 38, 0.8)',
      borderWidth: 2,
      borderDash: [6, 6],
      label: {
        display: true,
        content: `2.5V at ${labels[minIndex] || minIndex}s`,
        position: 'start',
        backgroundColor: 'rgba(220, 38, 38, 0.8)',
        color: 'white',
        font: {
          size: 11,
          weight: 'bold'
        },
        padding: 6
      }
    };
  }
  
  return annotations;
};

const createChart = (ctx, labels, actual, predicted, title) => {
  const datasets = [
    {
      label: `${title} Predicted`,
      data: predicted,
      borderColor: "#f97316",
      backgroundColor: "rgba(249, 115, 22, 0.1)",
      tension: 0.3,
      pointRadius: predicted.length > 500 ? 0 : 3,
      borderWidth: 2,
    },
  ];
  
  // Add actual data line if provided
  if (actual && actual.length > 0) {
    datasets.unshift({
      label: `${title} Actual`,
      data: actual,
      borderColor: "#3b82f6",
      backgroundColor: "rgba(59, 130, 246, 0.1)",
      borderDash: [5, 5],
      tension: 0.3,
      pointRadius: actual.length > 500 ? 0 : 3,
      borderWidth: 2,
    });
  }
  
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: predicted.length > 1000 ? 0 : 750,
      },
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          enabled: true,
        },
        decimation: {
          enabled: true,
          algorithm: "lttb",
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Time (seconds)",
          },
          ticks: {
            maxTicksLimit: 10,
          },
        },
        y: {
          title: {
            display: true,
            text: title,
          },
        },
      },
    },
  });
};

// Create ensemble chart with uncertainty bands
const createEnsembleChart = (ctx, labels, median, min, max, title) => {
  // Combine median, min, max for threshold detection
  const allVoltageData = [...median, ...min, ...max].filter(v => v != null);
  const annotations = findVoltageThresholds(allVoltageData, labels, title);
  
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `${title} Uncertainty Range`,
          data: max,
          borderColor: "rgba(249, 115, 22, 0.3)",
          backgroundColor: "rgba(249, 115, 22, 0.2)",
          fill: "+1", // Fill to next dataset (creates band)
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 1,
          borderDash: [5, 5],
        },
        {
          label: `${title} Min`,
          data: min,
          borderColor: "rgba(249, 115, 22, 0.3)",
          backgroundColor: "rgba(249, 115, 22, 0.2)",
          fill: false,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 1,
          borderDash: [5, 5],
        },
        {
          label: `${title} Median (Ensemble)`,
          data: median,
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.1)",
          tension: 0.3,
          pointRadius: median.length > 500 ? 0 : 3,
          borderWidth: 3,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: median.length > 1000 ? 0 : 750,
      },
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          enabled: true,
          callbacks: {
            afterBody: function(context) {
              if (context.length > 0) {
                const idx = context[0].dataIndex;
                return [
                  `Uncertainty: ±${((max[idx] - min[idx]) / 2).toFixed(3)}`,
                  `Range: ${min[idx].toFixed(3)} - ${max[idx].toFixed(3)}`
                ];
              }
            }
          }
        },
        decimation: {
          enabled: true,
          algorithm: "lttb",
        },
        annotation: {
          annotations: annotations
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Time (seconds)",
          },
          ticks: {
            maxTicksLimit: 10,
          },
        },
        y: {
          title: {
            display: true,
            text: title,
          },
        },
      },
    },
  });
};

const updateSummary = (element, actual, predicted) => {
  if (!actual.length || !predicted.length) {
    element.textContent = "No data available.";
    return;
  }

  const count = Math.min(actual.length, predicted.length);
  
  // Calculate RMSE
  const mse = actual
    .slice(0, count)
    .reduce((acc, value, index) => acc + (value - predicted[index]) ** 2, 0) / count;
  const rmse = Math.sqrt(mse);
  
  // Calculate MAE
  const mae = actual
    .slice(0, count)
    .reduce((acc, value, index) => acc + Math.abs(value - predicted[index]), 0) / count;

  element.textContent = `Samples: ${count} · RMSE: ${rmse.toFixed(4)} · MAE: ${mae.toFixed(4)}`;
};

const destroyPredictCharts = () => {
  if (predictVoltageChart) {
    predictVoltageChart.destroy();
    predictVoltageChart = null;
  }
  if (predictTemperatureChart) {
    predictTemperatureChart.destroy();
    predictTemperatureChart = null;
  }
};

const destroyCompareCharts = () => {
  if (compareVoltageChart) {
    compareVoltageChart.destroy();
    compareVoltageChart = null;
  }
  if (compareTemperatureChart) {
    compareTemperatureChart.destroy();
    compareTemperatureChart = null;
  }
};

const destroyEnsembleCharts = () => {
  if (ensembleVoltageChart) {
    ensembleVoltageChart.destroy();
    ensembleVoltageChart = null;
  }
  if (ensembleTemperatureChart) {
    ensembleTemperatureChart.destroy();
    ensembleTemperatureChart = null;
  }
};

const downsample = (data, maxPoints = 1000) => {
  if (data.length <= maxPoints) return data;
  const step = Math.ceil(data.length / maxPoints);
  return data.filter((_, idx) => idx % step === 0);
};

const handleCsv = (text) => {
  updateStatus("Processing CSV file...", "ready");
  const compareVoltageSummary = document.getElementById("compareVoltageSummary");
  const compareTemperatureSummary = document.getElementById("compareTemperatureSummary");
  const compareDownloadVoltage = document.getElementById("compareDownloadVoltage");
  const compareDownloadTemperature = document.getElementById("compareDownloadTemperature");
  
  compareVoltageSummary.textContent = "Loading...";
  compareTemperatureSummary.textContent = "Loading...";
  compareDownloadVoltage.disabled = true;
  compareDownloadTemperature.disabled = true;

  setTimeout(() => {
    const { headers, rows } = parseCsv(text);
    const missing = requiredHeaders.filter((header) => !headers.includes(header));

    if (missing.length) {
      destroyCompareCharts();
      compareVoltageSummary.textContent = `Missing columns: ${missing.join(", ")}`;
      compareTemperatureSummary.textContent = "";
      updateStatus(`Error: Missing columns - ${missing.join(", ")}`, "ready");
      return;
    }

    const indexMap = getColumnIndexMap(headers);
    const voltageActual = buildSeries(rows, indexMap, "voltage_actual");
    const voltagePred = buildSeries(rows, indexMap, "voltage_median_pred");
    const tempActual = buildSeries(rows, indexMap, "temperature_actual");
    const tempPred = buildSeries(rows, indexMap, "temperature_median_pred");

    const maxLength = Math.max(
      voltageActual.length,
      voltagePred.length,
      tempActual.length,
      tempPred.length
    );

    // Limit to 150 data points
    const displayLimit = 150;
    const voltageActualPlot = voltageActual.slice(0, displayLimit);
    const voltagePredPlot = voltagePred.slice(0, displayLimit);
    const tempActualPlot = tempActual.slice(0, displayLimit);
    const tempPredPlot = tempPred.slice(0, displayLimit);

    // Create time labels (0s, 2s, 4s, 6s, ...)
    const labels = Array.from({ length: Math.max(voltageActualPlot.length, tempActualPlot.length) }, (_, idx) => `${idx*2}s`);

    destroyCompareCharts();

    const voltageCtx = document.getElementById("compareVoltageChart");
    const temperatureCtx = document.getElementById("compareTemperatureChart");

    compareVoltageChart = createChart(voltageCtx, labels, voltageActualPlot, voltagePredPlot, "Voltage");
    compareTemperatureChart = createChart(temperatureCtx, labels, tempActualPlot, tempPredPlot, "Temperature");

    updateSummary(compareVoltageSummary, voltageActual, voltagePred);
    updateSummary(compareTemperatureSummary, tempActual, tempPred);

    compareDownloadVoltage.disabled = false;
    compareDownloadTemperature.disabled = false;
    
    const displayedPoints = Math.min(maxLength, displayLimit);
    updateStatus(`CSV Uploaded - Displaying ${displayedPoints} of ${maxLength} samples (${displayedPoints*2}s duration)`, "complete");
  }, 50);
};

csvInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    handleCsv(String(e.target.result || ""));
  };
  reader.readAsText(file);
});

// Download buttons for Predict tab
document.getElementById("predictDownloadVoltage").addEventListener("click", () => {
  if (!predictVoltageChart) return;
  const link = document.createElement("a");
  link.href = predictVoltageChart.toBase64Image();
  link.download = "predict-voltage-chart.png";
  link.click();
});

document.getElementById("predictDownloadTemperature").addEventListener("click", () => {
  if (!predictTemperatureChart) return;
  const link = document.createElement("a");
  link.href = predictTemperatureChart.toBase64Image();
  link.download = "predict-temperature-chart.png";
  link.click();
});

// Download buttons for Compare tab
document.getElementById("compareDownloadVoltage").addEventListener("click", () => {
  if (!compareVoltageChart) return;
  const link = document.createElement("a");
  link.href = compareVoltageChart.toBase64Image();
  link.download = "compare-voltage-chart.png";
  link.click();
});

document.getElementById("compareDownloadTemperature").addEventListener("click", () => {
  if (!compareTemperatureChart) return;
  const link = document.createElement("a");
  link.href = compareTemperatureChart.toBase64Image();
  link.download = "compare-temperature-chart.png";
  link.click();
});

// ================================
// ADAPTIVE TUNING (COMPARE TAB)
// ================================

const adaptiveRequiredHeaders = ["current", "voltage_actual", "temperature_actual"];
let adaptiveData = null;
let adaptiveRetrainPayload = null;
let adaptiveTrainingPoll = null;

const adaptiveActualInput = document.getElementById("adaptiveActualInput");
const adaptiveFileInfo = document.getElementById("adaptiveFileInfo");
const adaptiveEvaluateBtn = document.getElementById("adaptiveEvaluateBtn");
const adaptiveQueueBtn = document.getElementById("adaptiveQueueBtn");
const adaptiveEvalStatus = document.getElementById("adaptiveEvalStatus");
const adaptiveTrainMoeBtn = document.getElementById("adaptiveTrainMoeBtn");
const adaptiveTrainEnsembleBtn = document.getElementById("adaptiveTrainEnsembleBtn");
const adaptiveTrainingStatus = document.getElementById("adaptiveTrainingStatus");

const setAdaptiveEvalStatus = (message, type = "") => {
  adaptiveEvalStatus.textContent = message;
  adaptiveEvalStatus.className = `adaptive-status${type ? ` ${type}` : ""}`;
};

const setAdaptiveTrainingStatus = (message, type = "") => {
  adaptiveTrainingStatus.textContent = message;
  adaptiveTrainingStatus.className = `training-status${type ? ` ${type}` : ""}`;
};

const setAdaptiveTrainButtons = (modelsToTrain = [], forceDisable = false) => {
  if (forceDisable) {
    adaptiveTrainMoeBtn.disabled = true;
    adaptiveTrainEnsembleBtn.disabled = true;
    return;
  }
  adaptiveTrainMoeBtn.disabled = !modelsToTrain.includes("moe");
  adaptiveTrainEnsembleBtn.disabled = !modelsToTrain.includes("ensemble");
};

const formatAdaptiveLoss = (lossValue) => {
  if (typeof lossValue === "number" && Number.isFinite(lossValue)) {
    return lossValue.toFixed(6);
  }
  return "n/a";
};

const calculateMAPE = (actual, predicted) => {
  const count = Math.min(actual.length, predicted.length);
  if (!count) return NaN;
  let total = 0;
  for (let i = 0; i < count; i += 1) {
    const denom = Math.max(Math.abs(actual[i]), 1e-6);
    total += Math.abs((actual[i] - predicted[i]) / denom);
  }
  return (total / count) * 100;
};

const calculateMAE = (actual, predicted) => {
  const count = Math.min(actual.length, predicted.length);
  if (!count) return NaN;
  let total = 0;
  for (let i = 0; i < count; i += 1) {
    total += Math.abs(actual[i] - predicted[i]);
  }
  return total / count;
};

const stopAdaptiveTrainingPoll = () => {
  if (adaptiveTrainingPoll) {
    clearInterval(adaptiveTrainingPoll);
    adaptiveTrainingPoll = null;
  }
};

const refreshAdaptiveTrainingStatus = async () => {
  try {
    const response = await fetch(apiUrl('/training_status'));
    if (!response.ok) {
      throw new Error("Unable to fetch training status");
    }

    const result = await response.json();
    const training = result.training || {};

    if (training.running) {
      setAdaptiveTrainingStatus(
        `Training ${String(training.model_name || "").toUpperCase()} (${training.tuning_mode || "full"})... ${training.processed_samples || 0}/${training.total_samples || 0} | loss: ${formatAdaptiveLoss(training.last_loss)}`,
        "running"
      );
      setAdaptiveTrainButtons([], true);
      return;
    }

    if (training.status === "completed") {
      setAdaptiveTrainingStatus(
        `Completed ${String(training.model_name || "").toUpperCase()} tuning. Loss: ${formatAdaptiveLoss(training.last_loss)}`,
        "success"
      );
      stopAdaptiveTrainingPoll();
      const models = adaptiveRetrainPayload?.models_to_train || [];
      setAdaptiveTrainButtons(models);
      return;
    }

    if (training.status === "failed") {
      setAdaptiveTrainingStatus(`Training failed: ${training.error || training.message || "unknown error"}`, "error");
      stopAdaptiveTrainingPoll();
      const models = adaptiveRetrainPayload?.models_to_train || [];
      setAdaptiveTrainButtons(models);
      return;
    }

    setAdaptiveTrainingStatus(`Training status: ${training.message || "idle"}`);
  } catch (error) {
    setAdaptiveTrainingStatus(`Training status error: ${error.message}`, "error");
    stopAdaptiveTrainingPoll();
    const models = adaptiveRetrainPayload?.models_to_train || [];
    setAdaptiveTrainButtons(models);
  }
};

const startAdaptiveTrainingPoll = () => {
  stopAdaptiveTrainingPoll();
  adaptiveTrainingPoll = setInterval(refreshAdaptiveTrainingStatus, 2000);
};

const queueAdaptiveRetrainingSamples = async (triggeredBy = "manual") => {
  if (!adaptiveRetrainPayload) {
    updateStatus("Evaluate with actual data before queueing.", "ready");
    return false;
  }

  if (adaptiveRetrainPayload.queued) {
    return true;
  }

  adaptiveQueueBtn.disabled = true;
  setAdaptiveTrainButtons([], true);

  try {
    const response = await fetch(apiUrl('/queue_retraining_sample'), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...adaptiveRetrainPayload,
        triggered_by: triggeredBy
      })
    });

    if (!response.ok) {
      throw new Error("Failed to queue retraining samples");
    }

    const result = await response.json();
    adaptiveRetrainPayload.queued = true;
    const models = result.models_to_train || adaptiveRetrainPayload.models_to_train || [];
    setAdaptiveTrainButtons(models);
    setAdaptiveEvalStatus(`Queued for tuning: ${models.join(" + ").toUpperCase() || "NONE"}`, "ok");
    updateStatus(result.message || "Queued retraining samples.", "complete");
    return true;
  } catch (error) {
    updateStatus(`Error queueing samples: ${error.message}`, "ready");
    setAdaptiveEvalStatus(`Queue failed: ${error.message}`, "alert");
    adaptiveQueueBtn.disabled = false;
    const models = adaptiveRetrainPayload?.models_to_train || [];
    setAdaptiveTrainButtons(models);
    return false;
  }
};

const startAdaptiveTraining = async (modelName) => {
  if (!adaptiveRetrainPayload) {
    updateStatus("Evaluate with actual data before starting tuning.", "ready");
    return;
  }

  if (!adaptiveRetrainPayload.models_to_train?.includes(modelName)) {
    updateStatus(`${modelName.toUpperCase()} did not cross threshold for current sample.`, "ready");
    return;
  }

  if (!adaptiveRetrainPayload.queued) {
    const queued = await queueAdaptiveRetrainingSamples("manual_before_train");
    if (!queued) return;
  }

  const epochs = parseInt(document.getElementById("adaptiveTrainEpochs").value, 10);
  const maxSamples = parseInt(document.getElementById("adaptiveTrainMaxSamples").value, 10);

  if (!Number.isFinite(epochs) || epochs < 1 || epochs > 20) {
    updateStatus("Training epochs must be between 1 and 20.", "ready");
    return;
  }
  if (!Number.isFinite(maxSamples) || maxSamples < 1 || maxSamples > 500) {
    updateStatus("Max queued samples must be between 1 and 500.", "ready");
    return;
  }

  try {
    setAdaptiveTrainButtons([], true);
    setAdaptiveTrainingStatus(`Starting ${modelName.toUpperCase()} tuning...`, "running");

    const learningRate = modelName === "ensemble" ? 0.001 : 0.0001;

    const body = {
      model_name: modelName,
      tuning_mode: modelName === "moe" ? "adapter" : "full",
      epochs,
      max_samples: maxSamples,
      learning_rate: learningRate,
      rank: 8,
      alpha: 16,
      dropout: 0.05,
      batch_size: 4,
      accumulation_steps: 2
    };

    const response = await fetch(apiUrl('/train_queued_model'), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    const result = await response.json();
    if (!response.ok || (result.status !== "started" && result.status !== "success")) {
      throw new Error(result.message || "Failed to start tuning");
    }

    updateStatus(result.message || `${modelName.toUpperCase()} tuning started.`, "live");
    startAdaptiveTrainingPoll();
    await refreshAdaptiveTrainingStatus();
  } catch (error) {
    updateStatus(`Training start error: ${error.message}`, "ready");
    setAdaptiveTrainingStatus(`Training start failed: ${error.message}`, "error");
    const models = adaptiveRetrainPayload?.models_to_train || [];
    setAdaptiveTrainButtons(models);
  }
};

const evaluateAdaptiveWithActualData = async () => {
  if (!adaptiveData) {
    updateStatus("Upload actual CSV first.", "ready");
    return;
  }

  let soh = parseFloat(document.getElementById("adaptiveSOH").value);
  let initialVoltage = parseFloat(document.getElementById("adaptiveInitialVoltage").value);
  let initialTemperature = parseFloat(document.getElementById("adaptiveInitialTemperature").value);

  if (!Number.isFinite(soh)) soh = 0.95;
  if (!Number.isFinite(initialVoltage)) initialVoltage = adaptiveData.voltageActual[0];
  if (!Number.isFinite(initialTemperature)) initialTemperature = adaptiveData.temperatureActual[0];

  const steps = Math.min(adaptiveData.current.length, adaptiveData.voltageActual.length, adaptiveData.temperatureActual.length, 75);
  const currentDataSliced = adaptiveData.current.slice(0, steps);
  const actualVoltage = adaptiveData.voltageActual.slice(0, steps);
  const actualTemperature = adaptiveData.temperatureActual.slice(0, steps);

  const moeVoltageThreshold = parseFloat(document.getElementById("adaptiveMoeVoltageThreshold").value);
  const moeTempThreshold = parseFloat(document.getElementById("adaptiveMoeTempThreshold").value);
  const ensembleVoltageThreshold = parseFloat(document.getElementById("adaptiveEnsembleVoltageThreshold").value);
  const ensembleTempThreshold = parseFloat(document.getElementById("adaptiveEnsembleTempThreshold").value);
  const autoQueue = document.getElementById("adaptiveAutoQueue").checked;

  adaptiveEvaluateBtn.disabled = true;
  adaptiveQueueBtn.disabled = true;
  setAdaptiveTrainButtons([], true);
  setAdaptiveEvalStatus("Running model evaluation against actual data...");
  setAdaptiveTrainingStatus("Training status: idle");

  try {
    const transformerResponse = await fetch(apiUrl('/predict'), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        soh,
        voltage: initialVoltage,
        temperature: initialTemperature,
        current_data: currentDataSliced,
        steps
      })
    });

    if (!transformerResponse.ok) {
      throw new Error("MoE prediction failed for adaptive evaluation");
    }
    const transformerResult = await transformerResponse.json();

    const relativeAge = 1 - soh;
    const ensembleResponse = await fetch(apiUrl('/predict_ensemble'), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        relative_age: relativeAge,
        voltage: initialVoltage,
        temperature: initialTemperature,
        current_data: currentDataSliced,
        steps
      })
    });

    if (!ensembleResponse.ok) {
      throw new Error("Ensemble prediction failed for adaptive evaluation");
    }
    const ensembleResult = await ensembleResponse.json();

    const moeVoltage = transformerResult.voltage_forecast.slice(0, steps);
    const moeTemperature = transformerResult.temperature_forecast.slice(0, steps);
    const ensembleVoltage = ensembleResult.voltage_forecast.slice(0, steps);
    const ensembleTemperature = ensembleResult.temperature_forecast.slice(0, steps);

    const moeVoltageMAPE = calculateMAPE(actualVoltage, moeVoltage);
    const moeTempMAE = calculateMAE(actualTemperature, moeTemperature);
    const ensembleVoltageMAPE = calculateMAPE(actualVoltage, ensembleVoltage);
    const ensembleTempMAE = calculateMAE(actualTemperature, ensembleTemperature);

    const exceeded = {
      moe: {
        voltage_mape: moeVoltageMAPE > moeVoltageThreshold,
        temp_mae: moeTempMAE > moeTempThreshold
      },
      ensemble: {
        voltage_mape: ensembleVoltageMAPE > ensembleVoltageThreshold,
        temp_mae: ensembleTempMAE > ensembleTempThreshold
      }
    };

    const modelsToTrain = [];
    if (exceeded.moe.voltage_mape || exceeded.moe.temp_mae) modelsToTrain.push("moe");
    if (exceeded.ensemble.voltage_mape || exceeded.ensemble.temp_mae) modelsToTrain.push("ensemble");

    adaptiveRetrainPayload = {
      start_index: -1,
      sequence_length: steps,
      thresholds: {
        moe: { voltage_mape: moeVoltageThreshold, temp_mae: moeTempThreshold },
        ensemble: { voltage_mape: ensembleVoltageThreshold, temp_mae: ensembleTempThreshold }
      },
      metrics: {
        moe_voltage_mape: moeVoltageMAPE,
        moe_temp_mae: moeTempMAE,
        ensemble_voltage_mape: ensembleVoltageMAPE,
        ensemble_temp_mae: ensembleTempMAE
      },
      exceeded,
      models_to_train: modelsToTrain,
      parameters: {
        sequence_length: steps,
        soh,
        relative_age: relativeAge,
        initial_voltage: initialVoltage,
        initial_temperature: initialTemperature,
        current_profile: currentDataSliced,
        source: "user_actual_csv"
      },
      actual: {
        voltage: actualVoltage,
        temperature: actualTemperature
      },
      moe: {
        voltage: moeVoltage,
        temperature: moeTemperature,
        voltage_mape: moeVoltageMAPE,
        temp_mae: moeTempMAE
      },
      ensemble: {
        voltage: ensembleVoltage,
        temperature: ensembleTemperature,
        voltage_mape: ensembleVoltageMAPE,
        temp_mae: ensembleTempMAE
      },
      queued: false
    };

    const labels = Array.from({ length: steps }, (_, idx) => `${idx}s`);

    if (compareVoltageChart) compareVoltageChart.destroy();
    if (compareTemperatureChart) compareTemperatureChart.destroy();

    const compareVoltageCtx = document.getElementById("compareVoltageChart");
    compareVoltageChart = new Chart(compareVoltageCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "⚫ Actual", data: actualVoltage, borderColor: "#000000", borderWidth: 3, tension: 0.1, pointRadius: 2 },
          { label: "🟢 MoE", data: moeVoltage, borderColor: "#48bb78", borderWidth: 2, tension: 0.25, pointRadius: 1 },
          { label: "🔵 Ensemble", data: ensembleVoltage, borderColor: "#3b82f6", borderWidth: 2, tension: 0.25, pointRadius: 1 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false } }
    });

    const compareTempCtx = document.getElementById("compareTemperatureChart");
    compareTemperatureChart = new Chart(compareTempCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "⚫ Actual", data: actualTemperature, borderColor: "#000000", borderWidth: 3, tension: 0.1, pointRadius: 2 },
          { label: "🟢 MoE", data: moeTemperature, borderColor: "#48bb78", borderWidth: 2, tension: 0.25, pointRadius: 1 },
          { label: "🔵 Ensemble", data: ensembleTemperature, borderColor: "#3b82f6", borderWidth: 2, tension: 0.25, pointRadius: 1 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false } }
    });

    document.getElementById("compareVoltageSummary").textContent = `Actual vs Models: MoE ${moeVoltageMAPE.toFixed(3)}% | Ensemble ${ensembleVoltageMAPE.toFixed(3)}% (MAPE)`;
    document.getElementById("compareTemperatureSummary").textContent = `Actual vs Models: MoE ${moeTempMAE.toFixed(3)}°C | Ensemble ${ensembleTempMAE.toFixed(3)}°C (MAE)`;
    document.getElementById("compareDownloadVoltage").disabled = false;
    document.getElementById("compareDownloadTemperature").disabled = false;

    if (modelsToTrain.length > 0) {
      const reasons = [];
      if (exceeded.moe.voltage_mape) reasons.push(`MoE V MAPE ${moeVoltageMAPE.toFixed(3)}% > ${moeVoltageThreshold.toFixed(3)}%`);
      if (exceeded.moe.temp_mae) reasons.push(`MoE T MAE ${moeTempMAE.toFixed(3)}°C > ${moeTempThreshold.toFixed(3)}°C`);
      if (exceeded.ensemble.voltage_mape) reasons.push(`Ens V MAPE ${ensembleVoltageMAPE.toFixed(3)}% > ${ensembleVoltageThreshold.toFixed(3)}%`);
      if (exceeded.ensemble.temp_mae) reasons.push(`Ens T MAE ${ensembleTempMAE.toFixed(3)}°C > ${ensembleTempThreshold.toFixed(3)}°C`);

      setAdaptiveEvalStatus(`Threshold exceeded for ${modelsToTrain.join(" + ").toUpperCase()}: ${reasons.join(" | ")}`, "alert");
      adaptiveQueueBtn.disabled = false;

      if (autoQueue) {
        const queued = await queueAdaptiveRetrainingSamples("automatic_threshold");
        if (queued) {
          setAdaptiveTrainButtons(modelsToTrain);
        }
      }
    } else {
      setAdaptiveEvalStatus("Threshold not exceeded for MoE or Ensemble.", "ok");
      adaptiveQueueBtn.disabled = true;
      setAdaptiveTrainButtons([]);
    }

    updateStatus(`Adaptive evaluation complete for ${steps} steps.`, "complete");
  } catch (error) {
    updateStatus(`Adaptive evaluation error: ${error.message}`, "ready");
    setAdaptiveEvalStatus(`Evaluation failed: ${error.message}`, "alert");
    setAdaptiveTrainButtons([]);
    adaptiveQueueBtn.disabled = true;
  } finally {
    adaptiveEvaluateBtn.disabled = false;
  }
};

adaptiveActualInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  adaptiveData = null;
  adaptiveRetrainPayload = null;
  adaptiveQueueBtn.disabled = true;
  setAdaptiveTrainButtons([], true);

  if (!file) {
    adaptiveFileInfo.classList.remove("show");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const text = String(e.target.result || "");
    const { headers, rows } = parseCsv(text);
    const missing = adaptiveRequiredHeaders.filter((h) => !headers.includes(h));
    if (missing.length) {
      setAdaptiveEvalStatus(`CSV missing columns: ${missing.join(", ")}`, "alert");
      adaptiveFileInfo.classList.remove("show");
      return;
    }

    const idx = getColumnIndexMap(headers);
    const current = [];
    const voltageActual = [];
    const temperatureActual = [];

    rows.forEach((row) => {
      const c = toNumber(row[idx.current]);
      const v = toNumber(row[idx.voltage_actual]);
      const t = toNumber(row[idx.temperature_actual]);
      if (c !== null && v !== null && t !== null) {
        current.push(c);
        voltageActual.push(v);
        temperatureActual.push(t);
      }
    });

    if (current.length < 2) {
      setAdaptiveEvalStatus("CSV needs at least 2 valid rows.", "alert");
      adaptiveFileInfo.classList.remove("show");
      return;
    }

    adaptiveData = { current, voltageActual, temperatureActual };
    adaptiveFileInfo.innerHTML = `✓ Loaded ${current.length} rows (current + actual voltage + actual temperature)`;
    adaptiveFileInfo.classList.add("show");
    setAdaptiveEvalStatus("Data loaded. Click Evaluate vs Actual.");
    updateStatus(`Adaptive data loaded (${current.length} rows).`, "loaded");
  };
  reader.readAsText(file);
});

adaptiveEvaluateBtn.addEventListener("click", evaluateAdaptiveWithActualData);
adaptiveQueueBtn.addEventListener("click", async () => {
  await queueAdaptiveRetrainingSamples("manual");
});
adaptiveTrainMoeBtn.addEventListener("click", async () => {
  await startAdaptiveTraining("moe");
});
adaptiveTrainEnsembleBtn.addEventListener("click", async () => {
  await startAdaptiveTraining("ensemble");
});

// ================================
// ENSEMBLE TAB FUNCTIONALITY
// ================================

const ensembleCurrentInput = document.getElementById("ensembleCurrentInput");
const ensembleConstantCurrentInput = document.getElementById("ensembleConstantCurrent");
const ensembleCurrentStepsInput = document.getElementById("ensembleCurrentSteps");
const ensembleGenerateCurrentBtn = document.getElementById("ensembleGenerateCurrentBtn");
const ensemblePredictBtn = document.getElementById("ensemblePredictBtn");
const ensembleCurrentInfo = document.getElementById("ensembleCurrentInfo");

// Parameter inputs for ensemble
const ensembleAgeInput = document.getElementById("ensembleAge");
const ensembleVoltageInput = document.getElementById("ensembleVoltage");
const ensembleTemperatureInput = document.getElementById("ensembleTemperature");

let ensembleCurrentData = null;

// Ensemble current mode switching
document.querySelectorAll('input[name="ensembleCurrentMode"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    const mode = e.target.value;
    document.getElementById('ensembleConstantCurrentSection').style.display = mode === 'constant' ? 'block' : 'none';
    document.getElementById('ensembleCsvCurrentSection').style.display = mode === 'csv' ? 'block' : 'none';
    
    // Reset current data
    ensembleCurrentData = null;
    ensembleCurrentInfo.classList.remove('show');
    ensemblePredictBtn.disabled = mode === 'csv';
  });
});

// Generate constant current profile for ensemble
ensembleGenerateCurrentBtn.addEventListener('click', () => {
  const current = parseFloat(ensembleConstantCurrentInput.value);
  const steps = parseInt(ensembleCurrentStepsInput.value);
  
  ensembleCurrentData = new Array(steps).fill(current);
  
  ensembleCurrentInfo.innerHTML = `✓ Generated ${steps} constant current values (${current}A for ${steps}s)`;
  ensembleCurrentInfo.classList.add('show');
  ensemblePredictBtn.disabled = false;
  updateStatus(`Ensemble current profile ready - ${steps} steps at ${current}A`, 'loaded');
});

// Handle ensemble current CSV upload
ensembleCurrentInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    ensembleCurrentData = null;
    ensemblePredictBtn.disabled = true;
    ensembleCurrentInfo.classList.remove('show');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const text = String(e.target.result || "");
    const { headers, rows } = parseCsv(text);
    
    if (!headers.includes('current')) {
      updateStatus('Error: CSV must contain a "current" column', 'ready');
      ensembleCurrentData = null;
      ensemblePredictBtn.disabled = true;
      ensembleCurrentInfo.classList.remove('show');
      return;
    }
    
    const currentIndex = headers.indexOf('current');
    ensembleCurrentData = rows.map(row => parseFloat(row[currentIndex])).filter(val => !isNaN(val));
    
    // Limit to 75 steps for ensemble
    if (ensembleCurrentData.length > 75) {
      ensembleCurrentData = ensembleCurrentData.slice(0, 75);
    }
    
    ensembleCurrentInfo.innerHTML = `✓ Loaded ${ensembleCurrentData.length} current values (${ensembleCurrentData.length}s duration)`;
    ensembleCurrentInfo.classList.add('show');
    ensemblePredictBtn.disabled = false;
    updateStatus(`Ensemble current data loaded - ${ensembleCurrentData.length} samples ready`, 'loaded');
  };
  reader.readAsText(file);
});

// Ensemble prediction function
const predictWithEnsemble = async () => {
  if (!ensembleCurrentData || ensembleCurrentData.length === 0) {
    updateStatus('Error: Please upload current data first', 'ready');
    return;
  }

  const relativeAge = parseFloat(ensembleAgeInput.value);
  const voltage = parseFloat(ensembleVoltageInput.value);
  const temperature = parseFloat(ensembleTemperatureInput.value);
  const steps = ensembleCurrentData.length;

  updateStatus(`Generating ensemble forecast for ${steps} steps...`, "live");
  ensemblePredictBtn.disabled = true;

  try {
    // Call Python backend API
    const response = await fetch(apiUrl('/predict_ensemble'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        relative_age: relativeAge,
        voltage,
        temperature,
        current_data: ensembleCurrentData,
        steps
      })
    });

    if (!response.ok) {
      throw new Error('Ensemble prediction failed');
    }

    const result = await response.json();

    if (result.status === 'success') {
      const voltageForecast = result.voltage_forecast;
      const temperatureForecast = result.temperature_forecast;
      const voltageEnsemble = result.voltage_ensemble; // 10 model predictions
      const temperatureEnsemble = result.temperature_ensemble; // 10 model predictions

      // Calculate min/max for uncertainty bands
      const voltageMin = [];
      const voltageMax = [];
      const temperatureMin = [];
      const temperatureMax = [];

      for (let i = 0; i < steps; i++) {
        const vValues = voltageEnsemble.map(model => model[i]);
        const tValues = temperatureEnsemble.map(model => model[i]);
        
        voltageMin.push(Math.min(...vValues));
        voltageMax.push(Math.max(...vValues));
        temperatureMin.push(Math.min(...tValues));
        temperatureMax.push(Math.max(...tValues));
      }

      // Create time labels (2 seconds per step)
      const labels = Array.from({ length: steps }, (_, idx) => `${idx*2}s`);

      destroyEnsembleCharts();

      const voltageCtx = document.getElementById("ensembleVoltageChart");
      const temperatureCtx = document.getElementById("ensembleTemperatureChart");

      // Show forecast with uncertainty bands
      ensembleVoltageChart = createEnsembleChart(voltageCtx, labels, voltageForecast, voltageMin, voltageMax, "Voltage");
      ensembleTemperatureChart = createEnsembleChart(temperatureCtx, labels, temperatureForecast, temperatureMin, temperatureMax, "Temperature");

      // Calculate average uncertainty for summary
      const avgVoltageUncertainty = voltageMax.reduce((sum, max, i) => sum + (max - voltageMin[i]), 0) / steps;
      const avgTempUncertainty = temperatureMax.reduce((sum, max, i) => sum + (max - temperatureMin[i]), 0) / steps;

      document.getElementById("ensembleVoltageSummary").textContent = `Ensemble Forecast: ${steps} steps (${steps*2}s) | Avg Uncertainty: ±${(avgVoltageUncertainty / 2).toFixed(3)}V`;
      document.getElementById("ensembleTemperatureSummary").textContent = `Ensemble Forecast: ${steps} steps (${steps*2}s) | Avg Uncertainty: ±${(avgTempUncertainty / 2).toFixed(2)}°C`;

      document.getElementById("ensembleDownloadVoltage").disabled = false;
      document.getElementById("ensembleDownloadTemperature").disabled = false;

      updateStatus(`Ensemble Forecast Complete - ${steps} steps (${steps*2}s) | Age: ${relativeAge}`, "complete");
    } else {
      throw new Error(result.message || 'Unknown error');
    }

  } catch (error) {
    console.error('Ensemble prediction error:', error);
    updateStatus(`Error: ${error.message}. Make sure backend server is running on port 5000.`, "ready");
  } finally {
    ensemblePredictBtn.disabled = ensembleCurrentData === null;
  }
};

ensemblePredictBtn.addEventListener("click", predictWithEnsemble);

// Download buttons for Ensemble tab
document.getElementById("ensembleDownloadVoltage").addEventListener("click", () => {
  if (!ensembleVoltageChart) return;
  const link = document.createElement("a");
  link.href = ensembleVoltageChart.toBase64Image();
  link.download = "ensemble-voltage-chart.png";
  link.click();
});

document.getElementById("ensembleDownloadTemperature").addEventListener("click", () => {
  if (!ensembleTemperatureChart) return;
  const link = document.createElement("a");
  link.href = ensembleTemperatureChart.toBase64Image();
  link.download = "ensemble-temperature-chart.png";
  link.click();
});

// ================================
// MODEL COMPARISON TAB FUNCTIONALITY
// ================================

let mcCurrentData = null;
let mcVoltageChart = null;
let mcTemperatureChart = null;

// Current mode switching
document.querySelectorAll('input[name="mcCurrentMode"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    const mode = e.target.value;
    document.getElementById('mcConstantCurrentSection').style.display = mode === 'constant' ? 'block' : 'none';
    document.getElementById('mcCSVCurrentSection').style.display = mode === 'csv' ? 'block' : 'none';
    mcCurrentData = null;
    document.getElementById('mcCurrentInfo').classList.remove('show');
    document.getElementById('mcCompareBtn').disabled = mode === 'csv';
  });
});

// Generate constant current profile
document.getElementById('mcGenerateCurrentBtn').addEventListener('click', () => {
  const current = parseFloat(document.getElementById('mcConstantCurrent').value);
  const steps = parseInt(document.getElementById('mcCurrentSteps').value);
  
  mcCurrentData = Array(steps).fill(current);
  
  const infoBox = document.getElementById('mcCurrentInfo');
  const infoText = document.getElementById('mcCurrentInfoText');
  infoText.textContent = `✓ Generated ${steps} steps of ${current}A constant current`;
  infoBox.classList.add('show');
  document.getElementById('mcCompareBtn').disabled = false;
});

// CSV upload
document.getElementById('mcCurrentInput').addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target.result;
    const { headers, rows } = parseCsv(text);
    
    if (!headers.includes('current')) {
      updateStatus('Error: CSV must contain a "current" column', 'ready');
      return;
    }
    
    const currentIndex = headers.indexOf('current');
    mcCurrentData = rows.map(row => parseFloat(row[currentIndex])).filter(v => !isNaN(v)).slice(0, 75);
    
    const infoBox = document.getElementById('mcCurrentInfo');
    const infoText = document.getElementById('mcCurrentInfoText');
    infoText.textContent = `✓ Loaded ${mcCurrentData.length} current values from CSV`;
    infoBox.classList.add('show');
    document.getElementById('mcCompareBtn').disabled = false;
  };
  reader.readAsText(file);
});

// Main comparison function
const compareModels = async () => {
  if (!mcCurrentData) {
    updateStatus('Error: Please generate or upload current data first', 'ready');
    return;
  }
  
  const soh = parseFloat(document.getElementById('mcSOH').value);
  const voltage = parseFloat(document.getElementById('mcVoltage').value);
  const temperature = parseFloat(document.getElementById('mcTemperature').value);
  const steps = Math.min(mcCurrentData.length, 75); // Ensemble max
  const currentDataSliced = mcCurrentData.slice(0, steps);
  
  updateStatus(`Comparing models... (${steps} steps)`, "loading");
  document.getElementById('mcCompareBtn').disabled = true;
  
  try {
    // Call Transformer model
    const transformerResponse = await fetch(apiUrl('/predict'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        soh,
        voltage,
        temperature,
        current_data: currentDataSliced,
        steps
      })
    });
    
    if (!transformerResponse.ok) throw new Error('Transformer prediction failed');
    const transformerResult = await transformerResponse.json();
    
    // Call DeepEnsemble model
    const relativeAge = 1 - soh;
    const ensembleResponse = await fetch(apiUrl('/predict_ensemble'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        relative_age: relativeAge,
        voltage,
        temperature,
        current_data: currentDataSliced,
        steps
      })
    });
    
    if (!ensembleResponse.ok) throw new Error('Ensemble prediction failed');
    const ensembleResult = await ensembleResponse.json();
    
    // Extract predictions
    const transformerVoltage = transformerResult.voltage_forecast;
    const transformerTemp = transformerResult.temperature_forecast;
    const ensembleVoltage = ensembleResult.voltage_forecast;
    const ensembleTemp = ensembleResult.temperature_forecast;
    
    // Create time labels
    const labels = Array.from({ length: steps }, (_, idx) => `${idx}s`);
    
    // Destroy old charts
    if (mcVoltageChart) mcVoltageChart.destroy();
    if (mcTemperatureChart) mcTemperatureChart.destroy();
    
    // Create comparison charts
    const voltageCtx = document.getElementById("mcVoltageChart");
    const temperatureCtx = document.getElementById("mcTemperatureChart");
    
    mcVoltageChart = createComparisonChart(voltageCtx, labels, transformerVoltage, ensembleVoltage, "Voltage");
    mcTemperatureChart = createComparisonChart(temperatureCtx, labels, transformerTemp, ensembleTemp, "Temperature");
    
    // Calculate prediction differences (for comparison scoring)
    const voltageDiff = transformerVoltage.map((v, i) => Math.abs(v - ensembleVoltage[i]));
    const tempDiff = transformerTemp.map((v, i) => Math.abs(v - ensembleTemp[i]));
    
const avgVoltageDiff = voltageDiff.reduce((a, b) => a + b, 0) / voltageDiff.length;
    const avgTempDiff = tempDiff.reduce((a, b) => a + b, 0) / tempDiff.length;
    
    // Update summaries
    document.getElementById("mcVoltageSummary").textContent = 
      `Transformer vs Ensemble: Avg difference ${avgVoltageDiff.toFixed(4)}V`;
    document.getElementById("mcTemperatureSummary").textContent = 
      `Transformer vs Ensemble: Avg difference ${avgTempDiff.toFixed(3)}°C`;
    
    // Show that comparison is done (no actual values to compare against)
    document.getElementById("mcResults").style.display = "block";
    document.getElementById("mcTransformerVoltageScore").textContent = "Prediction Complete";
    document.getElementById("mcEnsembleVoltageScore").textContent = "Prediction Complete";
    document.getElementById("mcTransformerTempScore").textContent = "Prediction Complete";
    document.getElementById("mcEnsembleTempScore").textContent = "Prediction Complete";
    document.getElementById("mcVoltageWinner").textContent = 
      `ℹ️ To determine accuracy, compare these predictions against actual measurements in Compare Mode tab`;
    document.getElementById("mcTempWinner").textContent = "";
    
    document.getElementById("mcDownloadVoltage").disabled = false;
    document.getElementById("mcDownloadTemperature").disabled = false;
    
    updateStatus(`Comparison Complete - Both models predicted ${steps} steps`, "complete");
    
  } catch (error) {
    console.error('Comparison error:', error);
    updateStatus(`Error: ${error.message}. Make sure backend server is running.`, "ready");
  } finally {
    document.getElementById('mcCompareBtn').disabled = false;
  }
};

document.getElementById('mcCompareBtn').addEventListener('click', compareModels);

// Create comparison chart with both model predictions
const createComparisonChart = (ctx, labels, transformerData, ensembleData, title) => {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `🔮 Transformer ${title}`,
          data: transformerData,
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.1)",
          tension: 0.3,
          pointRadius: transformerData.length > 500 ? 0 : 3,
          borderWidth: 2,
        },
        {
          label: `🎯 DeepEnsemble ${title}`,
          data: ensembleData,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          tension: 0.3,
          pointRadius: ensembleData.length > 500 ? 0 : 3,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: transformerData.length > 1000 ? 0 : 750,
      },
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          enabled: true,
          callbacks: {
            afterBody: function(context) {
              if (context.length >= 2) {
                const transformer = context[0].parsed.y;
                const ensemble = context[1].parsed.y;
                const diff = Math.abs(transformer - ensemble);
                return `Difference: ${diff.toFixed(4)}`;
              }
            }
          }
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Time (seconds)",
          },
          ticks: {
            maxTicksLimit: 10,
          },
        },
        y: {
          title: {
            display: true,
            text: title,
          },
        },
      },
    },
  });
};

// Download buttons
document.getElementById("mcDownloadVoltage").addEventListener("click", () => {
  if (!mcVoltageChart) return;
  const link = document.createElement("a");
  link.href = mcVoltageChart.toBase64Image();
  link.download = "model-comparison-voltage.png";
  link.click();
});

document.getElementById("mcDownloadTemperature").addEventListener("click", () => {
  if (!mcTemperatureChart) return;
  const link = document.createElement("a");
  link.href = mcTemperatureChart.toBase64Image();
  link.download = "model-comparison-temperature.png";
  link.click();
});

// DATASET COMPARISON FUNCTIONALITY
let datasetVoltageChart, datasetTemperatureChart;

const getDatasetSplitInfo = async (sequenceLength, requestedStartIndex) => {
  const response = await fetch(apiUrl('/dataset_split_info'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sequence_length: sequenceLength,
      requested_start_index: requestedStartIndex
    })
  });

  const result = await response.json();
  if (!response.ok || result.status !== 'success') {
    throw new Error(result.message || 'Failed to get dataset split info');
  }

  return result;
};

document.getElementById('datasetFindTestBtn').addEventListener('click', async () => {
  const startIndexInput = document.getElementById('datasetStartIndex');
  const seqInput = document.getElementById('datasetSequenceLength');

  const requestedStartIndex = parseInt(startIndexInput.value, 10);
  const sequenceLength = parseInt(seqInput.value, 10);

  try {
    updateStatus('Finding nearest valid test index...', 'loading');
    const splitInfoResult = await getDatasetSplitInfo(sequenceLength, requestedStartIndex);
    const split = splitInfoResult.split || {};
    startIndexInput.value = split.suggested_test_start;
    updateStatus(
      `Valid test index selected: ${split.suggested_test_start} (test count: ${split.test_count})`,
      'loaded'
    );
  } catch (error) {
    updateStatus(`Error: ${error.message}`, 'ready');
  }
});

document.getElementById("datasetCompareBtn").addEventListener("click", async () => {
  const startIndex = parseInt(document.getElementById("datasetStartIndex").value, 10);
  const sequenceLength = parseInt(document.getElementById("datasetSequenceLength").value, 10);
  const useTestSplit = document.getElementById('datasetUseTestSplit').checked;
  
  updateStatus(`Loading dataset segment from index ${startIndex}...`, "loading");
  document.getElementById("datasetCompareBtn").disabled = true;
  document.getElementById('datasetFindTestBtn').disabled = true;
  
  try {
    const response = await fetch(apiUrl('/compare_with_dataset'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start_index: startIndex,
        sequence_length: sequenceLength,
        use_test_split: useTestSplit
      })
    });

    const result = await response.json();
    if (!response.ok || result.status !== 'success') {
      throw new Error(result.message || 'Dataset comparison failed');
    }
    
    // Extract data
    const labels = Array.from({ length: result.actual.voltage.length }, (_, idx) => `${idx}s`);
    const actualVoltage = result.actual.voltage;
    const actualTemp = result.actual.temperature;
    const moeVoltage = result.moe.voltage;
    const moeTemp = result.moe.temperature;
    const ensembleVoltage = result.ensemble.voltage;
    const ensembleTemp = result.ensemble.temperature;
    
    // Show results card
    document.getElementById("datasetResultsCard").style.display = "block";
    document.getElementById("datasetMoeVoltageMAPE").textContent = result.moe.voltage_mape.toFixed(4) + "%";
    document.getElementById("datasetMoeTempMAE").textContent = result.moe.temp_mae.toFixed(4) + "°C";
    document.getElementById("datasetEnsembleVoltageMAPE").textContent = result.ensemble.voltage_mape.toFixed(4) + "%";
    document.getElementById("datasetEnsembleTempMAE").textContent = result.ensemble.temp_mae.toFixed(4) + "°C";
    
    // Determine winner
    const moeWins = result.moe.voltage_mape < result.ensemble.voltage_mape;
    document.getElementById("datasetWinnerMessage").textContent = moeWins 
      ? "⭐ MoE Transformer wins with better accuracy!"
      : "⭐ Deep Ensemble wins with better accuracy!";
    document.getElementById("datasetWinnerMessage").style.background = moeWins ? "#e8f5e9" : "#e3f2fd";
    
    // Destroy old charts
    if (datasetVoltageChart) datasetVoltageChart.destroy();
    if (datasetTemperatureChart) datasetTemperatureChart.destroy();
    
    // Create voltage chart
    const voltageCtx = document.getElementById("datasetVoltageChart");
    datasetVoltageChart = new Chart(voltageCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "⚫ Actual (KIT Dataset)",
            data: actualVoltage,
            borderColor: "#000000",
            backgroundColor: "rgba(0, 0, 0, 0.1)",
            pointRadius: 4,
            pointHoverRadius: 6,
            borderWidth: 3,
            tension: 0.1
          },
          {
            label: "🟢 MoE Transformer",
            data: moeVoltage,
            borderColor: "#48bb78",
            backgroundColor: "rgba(72, 187, 120, 0.1)",
            pointRadius: 2,
            pointHoverRadius: 4,
            borderWidth: 2,
            tension: 0.3
          },
          {
            label: "🔵 Deep Ensemble",
            data: ensembleVoltage,
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.1)",
            pointRadius: 2,
            pointHoverRadius: 4,
            borderWidth: 2,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top" },
          tooltip: { enabled: true }
        },
        scales: {
          x: { title: { display: true, text: "Time (seconds)" } },
          y: { title: { display: true, text: "Voltage (V)" } }
        }
      }
    });
    
    // Create temperature chart
    const tempCtx = document.getElementById("datasetTemperatureChart");
    datasetTemperatureChart = new Chart(tempCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "⚫ Actual (KIT Dataset)",
            data: actualTemp,
            borderColor: "#000000",
            backgroundColor: "rgba(0, 0, 0, 0.1)",
            pointRadius: 4,
            pointHoverRadius: 6,
            borderWidth: 3,
            tension: 0.1
          },
          {
            label: "🟢 MoE Transformer",
            data: moeTemp,
            borderColor: "#48bb78",
            backgroundColor: "rgba(72, 187, 120, 0.1)",
            pointRadius: 2,
            pointHoverRadius: 4,
            borderWidth: 2,
            tension: 0.3
          },
          {
            label: "🔵 Deep Ensemble",
            data: ensembleTemp,
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.1)",
            pointRadius: 2,
            pointHoverRadius: 4,
            borderWidth: 2,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top" },
          tooltip: { enabled: true }
        },
        scales: {
          x: { title: { display: true, text: "Time (seconds)" } },
          y: { title: { display: true, text: "Temperature (°C)" } }
        }
      }
    });
    
    // Update summaries
    const splitMeta = result.data_split || {};
    const resolvedIndex = splitMeta.resolved_start_index ?? startIndex;
    const requestedIndex = splitMeta.requested_start_index ?? startIndex;
    const usedSplitText = useTestSplit ? ' | test split mode' : '';

    document.getElementById("datasetVoltageSummary").textContent = 
      `Segment index ${resolvedIndex} (requested ${requestedIndex})${usedSplitText}: MoE ${result.moe.voltage_mape.toFixed(3)}% vs Ensemble ${result.ensemble.voltage_mape.toFixed(3)}%`;
    document.getElementById("datasetTemperatureSummary").textContent = 
      `Segment index ${resolvedIndex} (requested ${requestedIndex})${usedSplitText}: MoE ${result.moe.temp_mae.toFixed(3)}°C vs Ensemble ${result.ensemble.temp_mae.toFixed(3)}°C`;

    if (useTestSplit && Number.isFinite(resolvedIndex)) {
      document.getElementById('datasetStartIndex').value = resolvedIndex;
    }
    
    // Enable download buttons
    document.getElementById("datasetDownloadVoltage").disabled = false;
    document.getElementById("datasetDownloadTemperature").disabled = false;

    if (useTestSplit && resolvedIndex !== startIndex) {
      updateStatus(`Dataset comparison complete - requested ${startIndex}, resolved to valid test index ${resolvedIndex}`, "complete");
    } else {
      updateStatus(`Dataset comparison complete - Segment loaded from index ${resolvedIndex}`, "complete");
    }
    
  } catch (error) {
    console.error('Dataset comparison error:', error);
    updateStatus(`Error: ${error.message}. Make sure backend server is running.`, "ready");
  } finally {
    document.getElementById("datasetCompareBtn").disabled = false;
    document.getElementById('datasetFindTestBtn').disabled = false;
  }
});

// Download buttons for dataset comparison
document.getElementById("datasetDownloadVoltage").addEventListener("click", () => {
  if (!datasetVoltageChart) return;
  const link = document.createElement("a");
  link.href = datasetVoltageChart.toBase64Image();
  link.download = "dataset-comparison-voltage.png";
  link.click();
});

document.getElementById("datasetDownloadTemperature").addEventListener("click", () => {
  if (!datasetTemperatureChart) return;
  const link = document.createElement("a");
  link.href = datasetTemperatureChart.toBase64Image();
  link.download = "dataset-comparison-temperature.png";
  link.click();
});

// Handle URL parameters for auto-loading segments
window.addEventListener('DOMContentLoaded', () => {
  const urlParams = new URLSearchParams(window.location.search);
  const tab = urlParams.get('tab');
  const index = urlParams.get('index');
  
  if (tab === 'datasetCompare' && index) {
    // Switch to dataset comparison tab
    const tabBtn = document.querySelector('[data-tab="datasetCompare"]');
    if (tabBtn) {
      tabBtn.click();
      
      // Wait a bit for tab to load, then set the index and trigger load
      setTimeout(() => {
        const indexInput = document.getElementById('datasetStartIndex');
        if (indexInput) {
          indexInput.value = index;
          document.getElementById('datasetCompareBtn').click();
        }
      }, 300);
    }
  }
});
