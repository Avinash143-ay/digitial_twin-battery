# Model Explanation

## Overview
This application uses two different deep learning models for battery forecasting, each with different architectures and prediction strategies.

---

## 🔮 Predict Mode (Tab 1)

### Model Used
**Transformer Model** (`models/simulator_cpu.pth`)  
Architecture: `v9_rescaling_adaptive_TransformerModel3Decoder`

### Architecture Details
- **Hidden Dimensions**: 150
- **Attention Heads**: 20
- **Decoder Layers**: 3
- **Maximum Steps**: 150 timesteps (300 seconds)

### How It Works
1. **Attention Mechanism**: Uses multi-head self-attention to capture relationships between different timesteps in the battery data
2. **Parallel Processing**: Processes the entire sequence at once, predicting all future values simultaneously
3. **Global Context**: Can "look ahead" and understand long-term dependencies in the data
4. **Input Parameters**: SOH (State of Health), initial voltage, initial temperature, current profile

### Prediction Strategy
- **Parallel Prediction**: Generates all predictions in one forward pass through the network
- **Results**: Produces smoother, more stable predictions that capture long-term trends
- **Best For**: Long-term forecasting, understanding overall battery behavior patterns

---

## 🎯 Ensemble Predict (Tab 3)

### Model Used
**DeepEnsemble Model** (`models/digital_twin_simpler.pt`)  
Architecture: Ensemble of 10 independent feedforward neural networks

### Architecture Details
- **Number of Models**: 10 independent neural networks
- **Base Architecture**: Feedforward neural network (BaseModel)
  - Input normalization layer
  - Multiple hidden layers
  - Output: voltage, temperature, resistance predictions
- **Maximum Steps**: 75 timesteps (150 seconds)

### How It Works
1. **Input Normalization**:
   - Voltage ÷ 3
   - Temperature ÷ 30
   - Resistance × 10
   - Current ÷ 5

2. **Ensemble Prediction**:
   - All 10 models make independent predictions
   - Median value is selected from the 10 predictions
   - This provides robustness against outliers

3. **Autoregressive Approach**:
   - Predicts one timestep at a time
   - Uses the prediction from step N as input for step N+1
   - Accumulates predictions sequentially

4. **Input Parameters**: Relative Age (= 1 - SOH), initial voltage, initial temperature, resistance, current profile

### Prediction Strategy
- **Autoregressive**: Predicts step-by-step, feeding each prediction back as input
- **Results**: Shows more fluctuations and variations, captures uncertainty through ensemble
- **Best For**: Understanding prediction uncertainty, modeling realistic battery variations

---

## Key Differences

| Aspect | Transformer (Predict Mode) | DeepEnsemble (Ensemble Predict) |
|--------|---------------------------|--------------------------------|
| **Architecture** | Transformer with attention | 10 feedforward networks |
| **Prediction Method** | Parallel (all at once) | Autoregressive (step-by-step) |
| **Max Steps** | 150 (300s) | 75 (150s) |
| **Input Parameter** | SOH | Relative Age (1 - SOH) |
| **Uncertainty** | Single prediction | Ensemble median |
| **Predictions** | Smoother curves | More fluctuations |
| **Error Accumulation** | Low | Moderate (autoregressive) |

---

## RMSE and MAE Metrics

### What They Measure
- **RMSE (Root Mean Squared Error)**: 
  - Measures average prediction error
  - More sensitive to large errors
  - Units: Same as measured quantity (V for voltage, °C for temperature)
  - Lower is better

- **MAE (Mean Absolute Error)**:
  - Measures average absolute difference
  - Treats all errors equally
  - Units: Same as measured quantity
  - Lower is better

### How to Use
1. Enter your actual measured values in the text areas (comma-separated)
2. Run the prediction
3. Metrics will automatically appear showing model accuracy
4. Actual values will be overlaid on the charts as dashed lines

### Example Input
```
Voltage: 3.7, 3.68, 3.66, 3.64, 3.62
Temperature: 28, 28.5, 29, 29.2, 29.5
```

---

## Which Model is Better?

Both models have been trained on the same battery data but learned different patterns:

### Use Transformer When:
- ✅ You need long-term forecasts (up to 300s)
- ✅ You want stable, smooth predictions
- ✅ You're planning battery management strategies
- ✅ You need consistent trend predictions

### Use DeepEnsemble When:
- ✅ You want to understand prediction uncertainty
- ✅ You need shorter-term forecasts (up to 150s)
- ✅ You want realistic voltage/temperature variations
- ✅ You're validating against real measurements

### Validation Recommendation
To determine which model is better for your specific use case:
1. Enter actual measured values from your battery tests
2. Compare RMSE and MAE for both models
3. The model with lower RMSE/MAE is more accurate for your data
4. Consider both accuracy and prediction style (smooth vs. realistic)

---

## Technical Notes

### Current Convention
- **Positive values** = Discharge (battery providing power)
- **Negative values** = Charge (battery receiving power)
- Range: -1.5A to +1.5A

### Timestep
Each prediction step = 2 seconds

### SOH vs Relative Age
- **SOH** (State of Health): 1.0 = brand new, 0.0 = fully degraded
- **Relative Age**: 1 - SOH (0.0 = brand new, 1.0 = fully degraded)
- For a battery with SOH = 0.95:
  - Use SOH = 0.95 in Predict Mode
  - Use Relative Age = 0.05 in Ensemble Mode

---

## Model Training

Both models were trained on real battery cycling data from:
- NASA Battery Dataset
- Multiple charge/discharge cycles
- Various operating conditions
- Temperature and current variations

The models learned to predict voltage and temperature evolution based on battery health and operating conditions.
