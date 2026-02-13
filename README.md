# Lightweight Digital Twin Forecasting

A web-based battery forecasting system using two deep learning models: **Transformer** and **DeepEnsemble** for predicting voltage and temperature behavior.

## 🚀 Features

- **Dual Model Architecture**: Compare Transformer vs DeepEnsemble predictions
- **3-Tab Interface**: 
  - 🔮 **Predict Mode**: Transformer model predictions (up to 150 steps/300s)
  - 📊 **Compare Mode**: Compare actual vs predicted values
  - 🎯 **Ensemble Predict**: DeepEnsemble model with uncertainty estimation (up to 75 steps/150s)
- **Real-time Predictions**: Autoregressive forecasting with voltage constraints (2.4V - 4.2V)
- **Interactive Charts**: Visualize battery behavior over time
- **Flexible Current Input**: Constant current or CSV upload

## 📁 Project Structure

```
├── backend/
│   └── backend.py          # Flask API with both models
├── frontend/
│   ├── index.html          # 3-tab web interface
│   ├── app.js              # Frontend logic & charts
│   └── style.css           # Styling
├── models/
│   ├── simulator_cpu.pth   # Transformer model weights
│   └── digital_twin_simpler.pt  # DeepEnsemble model weights
├── data/
│   └── example_current.csv # Sample current profile
├── start_server.bat        # Quick start script (Windows)
├── MODEL_EXPLANATION.md    # Detailed model documentation
├── PROJECT_STRUCTURE.md    # Architecture guide
└── requirements.txt        # Python dependencies
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### 1. Clone Repository
```bash
git clone https://github.com/Avinash143-ay/digitial_twin-battery.git
cd digitial_twin-battery
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- Flask & flask-cors
- PyTorch
- NumPy

### 3. Add Model Weights
Place your trained model files in the `models/` folder:
- `simulator_cpu.pth` (Transformer model)
- `digital_twin_simpler.pt` (DeepEnsemble model)

## 🎯 Quick Start

### Option 1: Using Batch Script (Windows)
```bash
start_server.bat
```

### Option 2: Manual Start
```bash
# Start backend server
cd backend
python backend.py

# Server runs at http://localhost:5000
# Open frontend/index.html in browser
```

## 📖 Usage Guide

### 🔮 Predict Mode (Transformer Model)

**Model**: v9_rescaling_adaptive_TransformerModel3Decoder
- **Max Steps**: 150 (300 seconds)
- **Prediction Method**: Parallel (all steps at once)
- **Best for**: Long-term smooth predictions

**Steps**:
1. Set **SOH** (State of Health): 0.0 - 1.0
   - 1.0 = brand new battery
   - 0.95 = 95% health (typical)
2. Set **Initial Voltage**: 3.2 - 4.2V (e.g., 3.7V)
3. Set **Initial Temperature**: 0 - 60°C (e.g., 28°C)
4. **Current Profile**:
   - **Constant**: Set current value (-1.5 to 1.5A) and steps
   - **CSV Upload**: `current` column, one value per 2s step
5. Click **"Predict Battery Behavior"**

### 🎯 Ensemble Predict (DeepEnsemble Model)

**Model**: 10 independent neural networks with median aggregation
- **Max Steps**: 75 (150 seconds)
- **Prediction Method**: Autoregressive (step-by-step)
- **Best for**: Uncertainty estimation and realistic variations

**Steps**:
1. Set **Relative Age**: 0.0 - 1.0
   - **Relative Age = 1 - SOH**
   - 0.05 = healthy battery (SOH 0.95)
   - 0.95 = degraded battery (SOH 0.05)
2. Set **Initial Voltage**: 3.2 - 4.2V
3. Set **Initial Temperature**: 0 - 60°C
4. **Current Profile**: Same as Predict Mode
5. Click **"Predict with Ensemble"**

**Note**: Voltage predictions are constrained to realistic battery range (2.4V - 4.2V)

### 📊 Compare Mode

Upload CSV with actual vs predicted data:

**Required Columns**:
- `voltage_actual`
- `voltage_median_pred`
- `temperature_actual`
- `temperature_median_pred`

**Example**:
```csv
voltage_actual,voltage_median_pred,temperature_actual,temperature_median_pred
3.7,3.68,28.0,28.2
3.69,3.67,28.5,28.4
3.68,3.66,28.8,28.6
```

## 📝 CSV Format Examples

### Current Profile (for predictions)
```csv
current
0.5
0.5
0.5
0.5
```

- Positive values = discharge (battery providing power)
- Negative values = charge (battery receiving power)
- Range: -1.5A to 1.5A
- Each row = 2 seconds

## 🧠 Model Comparison

| Feature | Transformer (Predict) | DeepEnsemble (Ensemble) |
|---------|----------------------|-------------------------|
| **Architecture** | Multi-head attention | 10 feedforward networks |
| **Max Steps** | 150 (300s) | 75 (150s) |
| **Prediction** | Parallel | Autoregressive |
| **Input** | SOH | Relative Age (1-SOH) |
| **Uncertainty** | Single prediction | Ensemble median |
| **Best Use** | Long-term trends | Short-term with uncertainty |
| **Predictions** | Smoother curves | More fluctuations |

See [MODEL_EXPLANATION.md](MODEL_EXPLANATION.md) for detailed technical documentation.

## 🔑 Key Parameters

### Current Convention
- **Positive (+)**: Discharge (battery → load)
- **Negative (-)**: Charge (charger → battery)

### SOH vs Relative Age
- **SOH** (State of Health): 1.0 = new, 0.0 = dead
- **Relative Age**: 1 - SOH (0.0 = new, 1.0 = dead)

For a battery with 95% health:
- Use **SOH = 0.95** in Predict Mode
- Use **Relative Age = 0.05** in Ensemble Mode

### Time Steps
- Each step = **2 seconds**
- 75 steps = 150 seconds (2.5 minutes)
- 150 steps = 300 seconds (5 minutes)

## 🛠️ API Endpoints

### POST /predict (Transformer)
```json
{
  "soh": 0.95,
  "voltage": 3.7,
  "temperature": 28,
  "current_data": [0.5, 0.5, 0.5],
  "steps": 3
}
```

### POST /predict_ensemble (DeepEnsemble)
```json
{
  "relative_age": 0.05,
  "voltage": 3.7,
  "temperature": 28,
  "current_data": [0.5, 0.5, 0.5],
  "steps": 3
}
```

**Response**:
```json
{
  "status": "success",
  "voltage_forecast": [3.68, 3.66, 3.64],
  "temperature_forecast": [28.2, 28.4, 28.6]
}
```

## 📚 Documentation

- **[MODEL_EXPLANATION.md](MODEL_EXPLANATION.md)**: Detailed model architecture and theory
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Code organization guide
- **[predictions.ipynb](predictions.ipynb)**: Training and evaluation notebook

## ⚠️ Important Notes

1. **Voltage Constraints**: Predictions are clipped to 2.4V - 4.2V (realistic lithium-ion range)
2. **Model Weights**: Not included in repo due to size. Train using predictions.ipynb or contact maintainer
3. **Data Files**: Large battery datasets excluded (>100MB GitHub limit)
4. **CORS Enabled**: Backend allows cross-origin requests for frontend access

## 🐛 Troubleshooting

### Backend won't start
- Check Python version (3.8+)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Ensure model files exist in `models/` folder

### Predictions fail
- Verify backend server is running (http://localhost:5000)
- Check browser console for errors
- Ensure current data is valid (-1.5 to 1.5A)

### Charts not displaying
- Open browser developer tools
- Check for JavaScript errors
- Clear browser cache and reload


## 👨‍💻 Author

**Avinash Reddy Guda**
- GitHub: [@Avinash143-ay](https://github.com/Avinash143-ay)
- Repository: [digitial_twin-battery](https://github.com/Avinash143-ay/digitial_twin-battery)

## 🙏 Acknowledgments

- PyTorch framework
- Chart.js for visualization
- Flask for backend API
