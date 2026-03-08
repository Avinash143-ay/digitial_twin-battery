# 🆕 Model Comparison Feature - Quick Guide

## What This Does

The new **⚖️ Model Comparison** tab runs **BOTH models** (Transformer + DeepEnsemble) on the **same input** and shows you their predictions side-by-side!

---

## How to Use

### **Step 1: Open the Website**
```bash
cd backend
python backend.py
```
Then open `frontend/index.html` in your browser

### **Step 2: Go to "⚖️ Model Comparison" Tab**
It's the second tab in the navigation

### **Step 3: Set Input Parameters**
- **State of Health (SOH)**: 0.95 (typical)
- **Initial Voltage**: 3.7V
- **Initial Temperature**: 28°C

### **Step 4: Configure Current Profile**
**Option A - Constant Current:**
1. Select "Constant Current"
2. Set current value: 1.0A
3. Set duration: 50 steps
4. Click "Generate Current Profile"

**Option B - Upload CSV:**
1. Select "Upload CSV"
2. Upload a CSV with `current` column
3. System will load up to 75 steps

### **Step 5: Click "🚀 Compare Both Models"**

The system will:
- ✅ Run Transformer model
- ✅ Run DeepEnsemble model
- ✅ Show BOTH predictions on same charts
- ✅ Display prediction differences

---

## What You'll See

### **Charts:**
- **Orange line** 🔮 = Transformer predictions
- **Blue line** 🎯 = DeepEnsemble predictions
- Hover over points to see difference between models

### **Results:**
- Average voltage difference between models
- Average temperature difference between models
- Both predictions displayed simultaneously

---

## Interpreting Results

### **When predictions are SIMILAR:**
```
Avg difference: 0.0234V
```
✅ Both models agree → High confidence in predictions

### **When predictions are DIFFERENT:**
```
Avg difference: 0.1523V
```
⚠️ Models disagree → Check input conditions or consider uncertainty

### **Typical Differences:**
- **Voltage**: <0.05V = good agreement
- **Temperature**: <1.0°C = good agreement

---

## To Determine Which is More Accurate

### **You need actual measurements!**

1. Save both model predictions
2. Collect real battery data with same conditions
3. Go to **📊 Compare Mode** tab
4. Upload comparison CSV for each model:

**For Transformer:**
```csv
voltage_actual,voltage_median_pred,temperature_actual,temperature_median_pred
3.70,3.68,28.0,28.1
...
```

**For DeepEnsemble:**
```csv
voltage_actual,voltage_median_pred,temperature_actual,temperature_median_pred
3.70,3.72,28.0,28.3
...
```

5. Compare RMSE/MAE scores
6. **Lower RMSE/MAE = More accurate model!**

---

## Example Workflow

### **Test Case: Battery Discharge at 1A**

**Setup:**
- SOH: 0.95
- Initial: 3.7V, 28°C
- Current: 1.0A constant
- Steps: 50

**Run Comparison:**
1. Generate current profile
2. Click "Compare Both Models"
3. Wait ~1-2 seconds

**Results:**
```
Voltage:
  🔮 Transformer: 3.70V → 3.42V (smooth curve)
  🎯 Ensemble: 3.70V → 3.44V (slight fluctuations)
  Avg Difference: 0.0234V

Temperature:
  🔮 Transformer: 28.0°C → 29.8°C
  🎯 Ensemble: 28.0°C → 30.1°C
  Avg Difference: 0.312°C
```

**Interpretation:**
- Models show good agreement (small differences)
- Both predict similar voltage drop pattern
- Ensemble predicts slightly higher temperature rise

---

## Benefits

### **Before (Old System):**
- Tab 1: Run Transformer
- Tab 3: Run Ensemble
- Manually compare results
- No visual side-by-side

### **Now (New Feature):**
-⚖️ Tab 2: Run BOTH models at once
- Visual side-by-side comparison
- Automatic difference calculation
- Faster workflow!

---

## Tips

### **For Best Results:**
✅ Use same test conditions you used for training  
✅ Keep steps ≤75 (DeepEnsemble limit)  
✅ Test with different SOH values (0.9, 0.8, 0.7)  
✅ Try different current profiles (charge/discharge)

### **Typical Use Cases:**
1. **Model Selection**: Which model should I use for this battery?
2. **Validation**: Do both models agree on predictions?
3. **Uncertainty Analysis**: How much do predictions vary?
4. **Demonstration**: Show stakeholders both approaches

---

## Troubleshooting

### **Problem: Button is disabled**
**Solution**: Generate or upload current profile first

### **Problem: "Prediction failed" error**
**Solution**: Make sure backend server is running on port 5000

### **Problem: Only one line shows on chart**
**Solution**: Models might be predicting identical values (zoom in to see difference)

---

Happy Comparing! 🎉
