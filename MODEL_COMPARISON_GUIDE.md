# Model Comparison Guide

## How to Determine Which Model is More Accurate

### **Current Metrics Available:**
Your Compare Mode (Tab 2) already calculates:
- **RMSE (Root Mean Squared Error)**: Measures average prediction error
  - Formula: √(Σ(predicted - actual)²/n)
  - Lower is better
  - More sensitive to large errors

---

## **Step-by-Step Comparison Process**

### **Step 1: Prepare Test Data**
You need real battery measurements with:
- Initial conditions (SOH, voltage, temperature)
- Current profile (what current was applied)
- Actual voltage measurements (ground truth)
- Actual temperature measurements (ground truth)

### **Step 2: Run Transformer Prediction**
1. Go to Tab 1 (Predict Mode)
2. Enter test conditions:
   - SOH: 0.95
   - Initial Voltage: 3.7V
   - Initial Temperature: 28°C
   - Upload your test current profile CSV
3. Click "Predict Battery Behavior"
4. **Save predictions** (screenshot or download chart)

### **Step 3: Run DeepEnsemble Prediction**
1. Go to Tab 3 (Ensemble Predict)
2. Enter same test conditions:
   - Relative Age: 0.05 (= 1 - 0.95)
   - Initial Voltage: 3.7V
   - Initial Temperature: 28°C
   - Upload same current profile CSV
3. Click "Predict with Ensemble"
4. **Save predictions** (screenshot or download chart)

### **Step 4: Compare Against Actual Data**

Create TWO CSV files:

**transformer_comparison.csv:**
```csv
voltage_actual,voltage_median_pred,temperature_actual,temperature_median_pred
3.70,3.68,28.0,28.1
3.68,3.67,28.2,28.3
3.66,3.65,28.4,28.5
...
```

**ensemble_comparison.csv:**
```csv
voltage_actual,voltage_median_pred,temperature_actual,temperature_median_pred
3.70,3.72,28.0,28.3
3.68,3.69,28.2,28.4
3.66,3.67,28.4,28.6
...
```

### **Step 5: Calculate RMSE**
1. Upload transformer_comparison.csv to Tab 2
2. Note the RMSE scores (e.g., Voltage RMSE: 0.0234)
3. Upload ensemble_comparison.csv to Tab 2
4. Note the RMSE scores (e.g., Voltage RMSE: 0.0312)

### **Step 6: Determine Winner**
**Whichever model has LOWER RMSE is more accurate!**

Example:
```
Transformer:
  Voltage RMSE: 0.0234 V
  Temperature RMSE: 0.421 °C

DeepEnsemble:
  Voltage RMSE: 0.0312 V
  Temperature RMSE: 0.389 °C

Result: Transformer is better for voltage, Ensemble is better for temperature
```

---

## **Additional Metrics You Can Add**

### **MAE (Mean Absolute Error)**
Less sensitive to outliers than RMSE:
```javascript
const mae = actual
  .slice(0, count)
  .reduce((acc, value, index) => acc + Math.abs(value - predicted[index]), 0) / count;
```

### **MAPE (Mean Absolute Percentage Error)**
Shows percentage error:
```javascript
const mape = (actual
  .slice(0, count)
  .reduce((acc, value, index) => 
    acc + Math.abs((value - predicted[index]) / value), 0) / count) * 100;
```

### **R² Score (Coefficient of Determination)**
Measures how well model fits data (0-1, higher is better):
```javascript
const mean = actual.reduce((a, b) => a + b) / actual.length;
const ssTotal = actual.reduce((acc, val) => acc + (val - mean) ** 2, 0);
const ssResidual = actual.reduce((acc, val, i) => 
  acc + (val - predicted[i]) ** 2, 0);
const r2 = 1 - (ssResidual / ssTotal);
```

---

## **Quick Comparison Tips**

### **When Transformer is Better:**
✅ Long-term predictions (>100 steps)
✅ Smooth, stable trends
✅ When you need fast inference (<50ms)

### **When DeepEnsemble is Better:**
✅ Uncertainty quantification needed
✅ Short-term predictions (<75 steps)
✅ When you want confidence intervals

### **Both Together:**
🎯 Use Transformer for initial forecast
🎯 Use Ensemble for uncertainty bounds
🎯 Combine predictions (weighted average based on RMSE)

---

## **Validation Dataset Recommendation**

For rigorous comparison, use:
1. **Different battery cell** (not P065 used for training)
2. **Different charge/discharge patterns**
3. **Different temperature ranges**
4. **Battery at different ages** (SOH: 0.9, 0.8, 0.7)

Test each model on same validation set and compare RMSE/MAE scores.

---

## **Expected Results**

Based on your architectures:
- **Transformer** should excel at: Long sequences, parallel prediction, smooth trends
- **DeepEnsemble** should excel at: Uncertainty estimation, capturing variations

Neither is "always better" - it depends on your use case!
