# Battery Forecasting System

## Quick Start

### 1. Place Model Weights
Put your `simulator_cpu.pth` file in this folder

### 2. Install Dependencies
```bash
pip install flask flask-cors torch numpy
```

### 3. Start Backend Server
```bash
python backend.py
```
Server runs at http://localhost:5000

### 4. Start Frontend
```bash
python -m http.server 8000
```
Access at http://localhost:8000

## How to Use

### 🔮 Predict Mode (Main Feature)
1. **Set Initial Conditions**:
   - SOH: 0.95 (95% battery health)
   - Initial Voltage: 3.7V
   - Initial Temperature: 28°C

2. **Upload Current Data**:
   - Create/upload CSV with `current` column
   - Values in Amperes (-1.5 to 1.5)
   - Each row = one 2-second time step
   - Example: `example_current.csv`

3. **Click "Predict Battery Behavior"**
   - Model forecasts voltage & temperature
   - Shows predictions over time (2s intervals)

### 📊 Compare Mode
Upload CSV with: `voltage_actual`, `voltage_median_pred`, `temperature_actual`, `temperature_median_pred`

## CSV Format Examples

**Current Data (for Predict):**
```csv
current
0.5
0.6
0.7
0.8
0.9
1.0
```

**Comparison Data:**
```csv
voltage_actual,voltage_median_pred,temperature_actual,temperature_median_pred
3.7,3.68,28.0,28.2
3.69,3.67,28.5,28.4
```

## Model Details

- **Inputs**: [SOH, Voltage, Temperature] + Current sequence
- **Output**: Voltage & Temperature predictions
- **Max Steps**: 150 (300 seconds)
- **Time Resolution**: 2 seconds per step
