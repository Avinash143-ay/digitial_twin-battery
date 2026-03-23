# Battery Digital Twin - Dual AI Architecture

A web-based battery forecasting system using two state-of-the-art deep learning models: **MoE-Enhanced Transformer** and **Deep Ensemble** for accurate voltage and temperature prediction with uncertainty quantification.

## 🎯 Models Trained on KIT Battery Dataset

Both models are trained on the **same challenging dataset**:
- **Source**: [KIT Battery Aging Dataset](https://www.nature.com/articles/s41597-024-03831-x) (Nature Scientific Data 2024)
- **Cell**: P065 (18650 Li-ion, 2Ah capacity)
- **Conditions**: 0°C temperature, 10-100% SOC (extreme low-temperature aging)
- **Performance**: MoE Transformer achieves **0.158% voltage MAPE** 🏆

## 🚀 Features

- **Dual AI Architecture**: MoE-Enhanced Transformer (0.158% error) vs Deep Ensemble (0.3-0.4% error)
- **5-Tab Interface**: 
  - ⚡ **Predict Mode**: MoE Transformer predictions (up to 150 steps/300s)
  - 📊 **Dataset Comparison**: Compare models against real KIT P065 measurements ✨ NEW
  - 🎯 **Model Comparison**: Side-by-side Transformer vs Ensemble
  - 📈 **Compare Mode**: Upload your own actual vs predicted data
  - 🔬 **Ensemble Predict**: Deep Ensemble with uncertainty (up to 75 steps/150s)
- **Real Dataset Validation**: Load segments from 1.1GB KIT battery dataset
- **Pre-computed Segments**: 8 demonstration segments with saved predictions (instant loading!)
- **Uncertainty Quantification**: Both models provide confidence estimates
- **Interactive Charts**: Visualize battery behavior with Chart.js
- **Flexible Current Input**: Constant current or CSV upload

## 📁 Project Structure

```
├── backend/
│   └── backend.py                # Flask API with MoE + Ensemble models
├── frontend/
│   ├── index.html                # 5-tab web interface
│   ├── app.js                    # Frontend logic & Chart.js
│   ├── style.css                 # Styling
│   └── segments_preview.html     # Pre-computed segments viewer ✨ NEW
├── Digital_Twin/
│   ├── digital_twin_best.pt      # MoE Transformer (0.158% error) 🏆
│   └── logging_stats/            # Training metrics (val_loss, MAPE)
├── models/
│   └── digital_twin_simpler.pt   # Deep Ensemble (10 × 200K networks)
├── data/
│   ├── example_current.csv       # Sample current profile
│   └── cell_log_age_2s_P065_1_S01_C03/  # KIT dataset (1.1GB CSV)
├── saved_predictions/            # Pre-computed demonstration segments ✨ NEW
│   ├── segment_summary.json      # Overview of all saved segments
│   └── segment_*.json            # Full predictions for 8 segments
├── find_good_segments.py         # Scan dataset for demo segments ✨ NEW
├── load_saved_segment.py         # Quick loader for saved predictions ✨ NEW
├── start_server.bat              # Quick start script (Windows)
├── POSTER_A0_BATTERY_DIGITAL_TWIN.md  # A0 poster content
├── MODEL_EXPLANATION.md          # Detailed model documentation
└── requirements.txt              # Python dependencies
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
Model files should be in the correct directories:
- `Digital_Twin/digital_twin_best.pt` (MoE-Enhanced Transformer)
- `models/digital_twin_simpler.pt` (Deep Ensemble)

### 4. Download KIT Dataset (Optional)
For dataset comparison feature:
- Download from: https://www.nature.com/articles/s41597-024-03831-x
- Place `cell_log_age_2s_P065_1_S01_C03.csv` in `data/cell_log_age_2s_P065_1_S01_C03/`
- Or use pre-computed segments in `saved_predictions/` folder
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

# Se⚡ Predict Mode (MoE-Enhanced Transformer)

**Model**: Digital_Twin_v1 with Mixture of Experts
- **Architecture**: 3 cascaded MoE layers (1000+ experts each, top-20 routing)
- **Performance**: 0.158% voltage MAPE, 0.698% temp MAPE 🏆
- **Max Steps**: 150 (300 seconds)
- **Prediction Method**: Parallel (all steps at once)
- **Output**: Mean (μ) and std deviation (σ) for uncertainty
- **Best Relative Age**: 0.0 - 1.0
   - Relative Age = 1 - SOH
   - 0.05 = healthy battery (SOH 0.95)
   - 0.95 = degraded battery (SOH 0.05)
2. Set **Initial Voltage**: 3.2 - 4.2V (e.g., 3.7V)
3. Set **Initial Temperature**: 0 - 60°C (e.g., 28°C)
4. **Current Profile**:
   - **Constant**: Set current value (-1.5 to 1.5A) and steps
   - **CSV Upload**: `current` column, one value per 2s step
5. Click **"Predict Battery Behavior"**
🔬 Ensemble Predict (Deep Ensemble Model)

**Model**: 10 independent neural networks with median aggregation
- **Architecture**: 10 × 200K parameter networks = 2M total
- **Performance**: ~0.3-0.4% voltage MAPE (2-3× worse than MoE)
- **Max Steps**: 75 (150 seconds)
- **Prediction Method**: Autoregressive (step-by-step)
- **Output**: Median of 10 predictions (uncertainty from spread)
- **Best for**: Validation, robustness testing

**Steps**: Same as Predict Mode

**Note**: Both models constrain voltage to 2.4V - 4.2V (realistic Li-ion range)

### 🎯 Model Comparison

**Purpose**: Side-by-side comparison of MoE vs Ensemble on same inputs

**Steps**:
1. Set parameters (same as Predict Mode)
2. Click **"Compare Both Models"**
3. See both predictions overlaid with different colors

### 📈 Compare Mode (Custom Data Upload)

**Purpose**: Compare your own actual vs predicted data
   - **Black solid line**: Actual measured data from KIT sensors
   - **Green dashed line**: MoE Transformer predictions
   - **Blue dashed line**: Deep Ensemble predictions

**Quick Start**: Open `frontend/segments_preview.html` to see 8 pre-computed segments with metrics!
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
MoE Transformer 🏆 | Deep Ensemble |
|---------|-------------------|---------------|
| **Architecture** | 3 MoE layers (1000+ experts) + Transformer | 10 × 200K networks |
| **Parameters** | ~500K | 2M |
| **Voltage MAPE** | **0.158%** ⭐ | 0.3-0.4% |
| **Temp MAPE** | **0.698%** ⭐ | 0.8-1.2% |
| **Max Steps** | 150 (300s) | 75 (150s) |
| **Prediction** | Parallel | Autoregressive |
| **Input** | Relative Age | Relative Age |
| **Uncertainty** | μ/σ outputs | Ensemble spread |
| **Best Use** | Highest accuracy + uncertainty | Validation + robustness |
| **Win Rate** | **94.4%** (17/18 segments) | 5.6% |

**Winner**: MoE Transformer (2-3× better accuracy!) 🎉
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
- EaGET /health
Health check endpoint
```json
{"status": "healthy", "models": ["moe_transformer", "deep_ensemble"]}
```

### POST /predict (MoE Transformer)
```json
{
  "relative_age": 0.05,
  "voltage": 3.7,
  "temperature": 28,
  "current_data": [0.5, 0.5, 0.5],
  "steps": 3
}
```

### POST /predict_ensemble (Deep Ensemble)
```json
{
  "relative_age": 0.05,
  "voltage": 3.7,
  "temperature": 28,
  "current_data": [0.5, 0.5, 0.5],
  "steps": 3
}
```

### POST /compare_with_dataset (NEW! ✨)
Load KIT dataset segment and compare both models
```json
{🎯 Pre-computed Demonstration Segments

We've pre-scanned the KIT dataset and saved 8 excellent demonstration segments where MoE wins decisively:

| Index | MoE MAPE | Ensemble MAPE | Improvement |
|-------|----------|---------------|-------------|
| 10000 | 0.333% | 12.218% | **37× better!** |
| 90000 | 2.242% | 18.036% | **8× better!** |
| 25000 | 1.154% | 25.773% | **22× better!** |
| 30000 | 0.513% | 31.619% | **62× better!** |
| 35000 | 0.388% | 32.910% | **85× better!** |

**Quick Access**:
1. Open `frontend/segments_preview.html` to see all saved segments
2. Click any segment to auto-load it in the main dashboard
3. Or manually enter these indices in the Dataset Comparison tab

**Find More Segments**:
```bash
python find_good_segments.py  # Scans dataset, saves predictions
python load_saved_segment.py  # Loads and visualizes saved segments
```

## 📚 Documentation

- **[POSTER_A0_BATTERY_DIGITAL_TWIN.md](POSTER_A0_BATTERY_DIGITAL_TWIN.md)**: A0 poster content with all technical details
- **[MODEL_EXPLANATION.md](MODEL_EXPLANATION.md)**: Detailed model architecture and theory
- **[WHY_TWO_MODELS.md](WHY_TWO_MODELS.md)**: Rationale for dual-model architecture
}
```

**Response**:
```jsDataset**: Both models trained on KIT P065 (0°C, 10-100% SOC extreme conditions)
2. **Voltage Constraints**: Predictions clipped to 2.4V - 4.2V (realistic Li-ion range)
3. **Model Weights**: MoE model in `Digital_Twin/`, Ensemble in `models/`
4. **Large Files**: 1.1GB dataset not included (download from Nature Scientific Data)
5. **Pre-computed Segments**: Use `saved_predictions/` for instant demos without full dataset
6. **CORS Enabled**: Backend allows cross-origin requests for frontend access
7. **Input Format**: Both models now use `Relative_Age` (not SOH)
    "temperature": [0.0, 0.0, ...]
  },
  "moe": {
    "voltage": [3.81, 3.82, ...],
    "temperature": [0.0, 0.0, ...],
    "voltage_mape": 0.333,
    "temp_mae": 0.026
  },
  "ensemble": {
    "voltage": [3.37, 3.38, ...],
    "temperature": [0.14, 0.15, ...],
    "voltage_mape": 12.218, (F12)
- Check for JavaScript errors in Console tab
- Clear browser cache and reload
- Ensure Chart.js is loaded (check Network tab)

### Dataset comparison fails
- Verify KIT dataset exists at `data/cell_log_age_2s_P065_1_S01_C03/cell_log_age_2s_P065_1_S01_C03.csv`
- Or use pre-computed segments in `saved_predictions/` folder
- Check start_index is valid (> 0, < dataset length)

### Segments preview doesn't load
- Open `segments_preview.html` from `frontend/` folder
- Check that `saved_predictions/segment_summary.json` exists
- Run `python find_good_segments.py` to generate saved segments

## 🎓 Citation

If you use this code or the KIT dataset, please cite:

```bibtex
@article{kitdataset2024,
  title={KIT Battery Aging Dataset},
  journal={Nature Scientific Data},
  year={2024},
  doi={10.1038/s41597-024-03831-x},
  url={https://www.nature.com/articles/s41597-024-03831-x}
}
```  "parameters": {...}
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
