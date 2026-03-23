# Lightweight Digital Twin Forecasting - Project Structure

## Folder Organization

```
untitled folder/
├── models/                         # Trained model files
│   ├── simulator_cpu.pth          # Transformer model (Predict Mode)
│   └── digital_twin_simpler.pt    # DeepEnsemble model (Ensemble Predict)
│
├── backend/                        # Backend server
│   └── backend.py                 # Flask API server (both models)
│
├── frontend/                       # Frontend files
│   ├── index.html                 # Main web interface
│   ├── style.css                  # Styling
│   └── app.js                     # JavaScript logic
│
├── data/                          # Data files
│   ├── example_current.csv        # Example current profile
│   └── cell_log_age_2s_P065_1_S01_C03/  # Battery test data
│
├── predictions.ipynb              # Jupyter notebook (ensemble model development)
├── requirements.txt               # Python dependencies
└── start_server.bat              # Quick start script

```

## Models

### 1. Transformer Model (`simulator_cpu.pth`)
- **Tab**: Predict Mode
- **Architecture**: Multi-layer Transformer with attention mechanism
- **Input**: [SOH, Voltage, Temperature] + Current sequence
- **Output**: Voltage and Temperature predictions (up to 150 steps)
- **Method**: Parallel prediction (all steps at once)

### 2. DeepEnsemble Model (`digital_twin_simpler.pt`)
- **Tab**: Ensemble Predict
- **Architecture**: 10 feedforward neural networks
- **Input**: [Relative_Age, Voltage, Temperature, Current] per step
- **Output**: Median voltage and temperature from 10 models (up to 75 steps)
- **Method**: Autoregressive (each prediction feeds into next)

## How to Run

### Option 1: Quick Start
Double-click `start_server.bat`

### Option 2: Manual Start
1. Open terminal in `backend/` folder
2. Run: `python backend.py`
3. Open `frontend/index.html` in your browser

## API Endpoints

- `POST /predict` - Transformer model predictions
- `POST /predict_ensemble` - DeepEnsemble predictions
- `GET /health` - Server health check

## Model Differences

| Feature | Transformer (Tab 1) | DeepEnsemble (Tab 3) |
|---------|-------------------|---------------------|
| Input Parameter | SOH (0-1) | Relative Age (0-1) |
| Max Steps | 150 | 75 |
| Prediction Type | All at once | Autoregressive |
| Uncertainty | No | Yes (10 models) |
