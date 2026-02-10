const csvInput = document.getElementById("csvInput");
const currentInput = document.getElementById("currentInput");
const constantCurrentInput = document.getElementById("constantCurrent");
const currentStepsInput = document.getElementById("currentSteps");
const generateCurrentBtn = document.getElementById("generateCurrentBtn");
const voltageSummary = document.getElementById("voltageSummary");
const temperatureSummary = document.getElementById("temperatureSummary");
const downloadVoltage = document.getElementById("downloadVoltage");
const downloadTemperature = document.getElementById("downloadTemperature");
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

let voltageChart;
let temperatureChart;
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
    } else {
      updateStatus('Ready - Upload CSV for comparison', 'ready');
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

      destroyCharts();

      const voltageCtx = document.getElementById("voltageChart");
      const temperatureCtx = document.getElementById("temperatureChart");

      // Show forecast as predicted line
      voltageChart = createChart(voltageCtx, labels, [], voltageForecast, "Voltage");
      temperatureChart = createChart(temperatureCtx, labels, [], temperatureForecast, "Temperature");

      voltageSummary.textContent = `Forecast: ${steps} steps (${steps * 2}s)`;
      temperatureSummary.textContent = `Forecast: ${steps} steps (${steps * 2}s)`;

      downloadVoltage.disabled = false;
      downloadTemperature.disabled = false;

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

const destroyCharts = () => {
  if (voltageChart) {
    voltageChart.destroy();
    voltageChart = null;
  }
  if (temperatureChart) {
    temperatureChart.destroy();
    temperatureChart = null;
  }
};

const downsample = (data, maxPoints = 1000) => {
  if (data.length <= maxPoints) return data;
  const step = Math.ceil(data.length / maxPoints);
  return data.filter((_, idx) => idx % step === 0);
};

const handleCsv = (text) => {
  updateStatus("Processing CSV file...", "ready");
  voltageSummary.textContent = "Loading...";
  temperatureSummary.textContent = "Loading...";
  downloadVoltage.disabled = true;
  downloadTemperature.disabled = true;

  setTimeout(() => {
    const { headers, rows } = parseCsv(text);
    const missing = requiredHeaders.filter((header) => !headers.includes(header));

    if (missing.length) {
      destroyCharts();
      voltageSummary.textContent = `Missing columns: ${missing.join(", ")}`;
      temperatureSummary.textContent = "";
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

    destroyCharts();

    const voltageCtx = document.getElementById("voltageChart");
    const temperatureCtx = document.getElementById("temperatureChart");

    voltageChart = createChart(voltageCtx, labels, voltageActualPlot, voltagePredPlot, "Voltage");
    temperatureChart = createChart(temperatureCtx, labels, tempActualPlot, tempPredPlot, "Temperature");

    updateSummary(voltageSummary, voltageActual, voltagePred);
    updateSummary(temperatureSummary, tempActual, tempPred);

    downloadVoltage.disabled = false;
    downloadTemperature.disabled = false;
    
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

downloadVoltage.addEventListener("click", () => {
  if (!voltageChart) return;
  const link = document.createElement("a");
  link.href = voltageChart.toBase64Image();
  link.download = "voltage-chart.png";
  link.click();
});

downloadTemperature.addEventListener("click", () => {
  if (!temperatureChart) return;
  const link = document.createElement("a");
  link.href = temperatureChart.toBase64Image();
  link.download = "temperature-chart.png";
  link.click();
});
