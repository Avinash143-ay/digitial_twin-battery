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
  
  currentInfo.innerHTML = `✓ Generated ${steps} constant current values (${current}A for ${steps * 2}s)`;
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
    
    currentInfo.innerHTML = `✓ Loaded ${currentData.length} current values (${currentData.length * 2}s duration)`;
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
    const response = await fetch('http://localhost:5000/predict', {
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
      const labels = Array.from({ length: steps }, (_, idx) => `${idx * 2}s`);

      destroyPredictCharts();

      const voltageCtx = document.getElementById("predictVoltageChart");
      const temperatureCtx = document.getElementById("predictTemperatureChart");

      // Show forecast as predicted line
      predictVoltageChart = createChart(voltageCtx, labels, [], voltageForecast, "Voltage");
      predictTemperatureChart = createChart(temperatureCtx, labels, [], temperatureForecast, "Temperature");

      document.getElementById("predictVoltageSummary").textContent = `Forecast: ${steps} steps (${steps * 2}s)`;
      document.getElementById("predictTemperatureSummary").textContent = `Forecast: ${steps} steps (${steps * 2}s)`;

      document.getElementById("predictDownloadVoltage").disabled = false;
      document.getElementById("predictDownloadTemperature").disabled = false;

      updateStatus(`Forecast Complete - ${steps} steps (${steps * 2}s) | SOH: ${soh}`, "complete");
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

const parseCsv = (text) => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    return { headers: [], rows: [] };
  }

  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines.slice(1).map((line) => line.split(",").map((cell) => cell.trim()));

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

const createChart = (ctx, labels, actual, predicted, title) =>
  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `${title} Actual`,
          data: actual,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.1)",
          tension: 0.3,
          pointRadius: actual.length > 500 ? 0 : 3,
          borderWidth: 2,
        },
        {
          label: `${title} Predicted`,
          data: predicted,
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.1)",
          tension: 0.3,
          pointRadius: predicted.length > 500 ? 0 : 3,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: actual.length > 1000 ? 0 : 750,
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

const updateSummary = (element, actual, predicted) => {
  if (!actual.length || !predicted.length) {
    element.textContent = "No data available.";
    return;
  }

  const count = Math.min(actual.length, predicted.length);
  const mse = actual
    .slice(0, count)
    .reduce((acc, value, index) => acc + (value - predicted[index]) ** 2, 0) / count;
  const rmse = Math.sqrt(mse);

  element.textContent = `Samples: ${count} · RMSE: ${rmse.toFixed(4)}`;
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
    const labels = Array.from({ length: Math.max(voltageActualPlot.length, tempActualPlot.length) }, (_, idx) => `${idx * 2}s`);

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
    updateStatus(`CSV Uploaded - Displaying ${displayedPoints} of ${maxLength} samples (${displayedPoints * 2}s duration)`, "complete");
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
  
  ensembleCurrentInfo.innerHTML = `✓ Generated ${steps} constant current values (${current}A for ${steps * 2}s)`;
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
    
    ensembleCurrentInfo.innerHTML = `✓ Loaded ${ensembleCurrentData.length} current values (${ensembleCurrentData.length * 2}s duration)`;
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
    const response = await fetch('http://localhost:5000/predict_ensemble', {
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

      // Create time labels
      const labels = Array.from({ length: steps }, (_, idx) => `${idx * 2}s`);

      destroyEnsembleCharts();

      const voltageCtx = document.getElementById("ensembleVoltageChart");
      const temperatureCtx = document.getElementById("ensembleTemperatureChart");

      // Show forecast as predicted line
      ensembleVoltageChart = createChart(voltageCtx, labels, [], voltageForecast, "Voltage");
      ensembleTemperatureChart = createChart(temperatureCtx, labels, [], temperatureForecast, "Temperature");

      document.getElementById("ensembleVoltageSummary").textContent = `Ensemble Forecast: ${steps} steps (${steps * 2}s)`;
      document.getElementById("ensembleTemperatureSummary").textContent = `Ensemble Forecast: ${steps} steps (${steps * 2}s)`;

      document.getElementById("ensembleDownloadVoltage").disabled = false;
      document.getElementById("ensembleDownloadTemperature").disabled = false;

      updateStatus(`Ensemble Forecast Complete - ${steps} steps (${steps * 2}s) | Age: ${relativeAge}`, "complete");
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
