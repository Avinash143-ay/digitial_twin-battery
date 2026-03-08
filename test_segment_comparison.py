import pandas as pd
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

# Read a segment from the KIT dataset
print("Loading KIT dataset segment...")
dataset_path = 'data/cell_log_age_2s_P065_1_S01_C03/cell_log_age_2s_P065_1_S01_C03.csv'

# Read just a small chunk to test
# Let's start from row 10000 and take 150 steps (max sequence length)
START_INDEX = 10000
SEQUENCE_LENGTH = 75  # Use 75 for ensemble compatibility

# Read the specific segment
df = pd.read_csv(dataset_path, sep=';', skiprows=range(1, START_INDEX), nrows=SEQUENCE_LENGTH+1)
print(f"\nLoaded segment from index {START_INDEX} to {START_INDEX + SEQUENCE_LENGTH}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Extract initial conditions from first row
initial_row = df.iloc[0]
print(f"\n=== Initial Conditions (Index {START_INDEX}) ===")
print(f"Available columns: {df.columns.tolist()}")

# KIT dataset column names
initial_voltage = float(initial_row['v_raw_V'])
voltage_actual = df['v_raw_V'].values[1:SEQUENCE_LENGTH+1]

initial_temp = float(initial_row['t_cell_degC'])
temp_actual = df['t_cell_degC'].values[1:SEQUENCE_LENGTH+1]

current_profile = df['i_raw_A'].values[1:SEQUENCE_LENGTH+1]  # Already in Amperes

# Get SOC if available
if 'soc_est' in df.columns:
    initial_soc = float(initial_row['soc_est'])
    print(f"Initial SOC: {initial_soc:.3f}")
    # Estimate relative age from capacity if available
    if 'cap_aged_est_Ah' in df.columns and not pd.isna(initial_row['cap_aged_est_Ah']):
        nominal_capacity = 2.0  # P065 is 2Ah nominal
        aged_capacity = float(initial_row['cap_aged_est_Ah'])
        soh = aged_capacity / nominal_capacity
        relative_age = 1.0 - soh
    else:
        # Estimate based on test conditions (0°C, later in cycle)
        relative_age = 0.35
        soh = 1.0 - relative_age
else:
    relative_age = 0.35
    soh = 1.0 - relative_age

# Estimate relative age (SOH) - this might not be directly in the data
# For now, use a reasonable value based on cell aging
relative_age = 0.35  # Assuming some aging (you can adjust this)
soh = 1.0 - relative_age

print(f"Initial Voltage: {initial_voltage:.4f} V")
print(f"Initial Temperature: {initial_temp:.2f} °C")
print(f"Estimated SOH: {soh:.2f} (Relative Age: {relative_age:.2f})")
print(f"Current profile length: {len(current_profile)}")
print(f"Current range: {current_profile.min():.4f} to {current_profile.max():.4f} A")

# Prepare data for models
print("\n=== Calling Backend API ===")
import requests
import json

API_URL = 'http://localhost:5000'

# Call MoE Transformer
print("\n1. MoE-Enhanced Transformer...")
moe_payload = {
    'soh': soh,
    'voltage': initial_voltage,
    'temperature': initial_temp,
    'current_data': current_profile.tolist()
}

try:
    moe_response = requests.post(f'{API_URL}/predict', json=moe_payload)
    moe_result = moe_response.json()
    moe_voltage_pred = np.array(moe_result['voltage_forecast'])
    moe_temp_pred = np.array(moe_result['temperature_forecast'])
    print(f"✓ MoE predictions received: {len(moe_voltage_pred)} steps")
except Exception as e:
    print(f"✗ MoE prediction failed: {e}")
    sys.exit(1)

# Call Deep Ensemble
print("\n2. Deep Ensemble...")
ensemble_payload = {
    'relative_age': relative_age,
    'voltage': initial_voltage,
    'temperature': initial_temp,
    'current_data': current_profile.tolist()
}

try:
    ensemble_response = requests.post(f'{API_URL}/predict_ensemble', json=ensemble_payload)
    ensemble_result = ensemble_response.json()
    ensemble_voltage_pred = np.array(ensemble_result['voltage_forecast'])
    ensemble_temp_pred = np.array(ensemble_result['temperature_forecast'])
    print(f"✓ Ensemble predictions received: {len(ensemble_voltage_pred)} steps")
except Exception as e:
    print(f"✗ Ensemble prediction failed: {e}")
    sys.exit(1)

# Calculate errors
print("\n=== COMPARISON RESULTS ===")

# Voltage MAPE
moe_voltage_mape = np.mean(np.abs((voltage_actual - moe_voltage_pred) / voltage_actual)) * 100
ensemble_voltage_mape = np.mean(np.abs((voltage_actual - ensemble_voltage_pred) / voltage_actual)) * 100

# Temperature MAE (MAPE not good for temp near 0)
moe_temp_mae = np.mean(np.abs(temp_actual - moe_temp_pred))
ensemble_temp_mae = np.mean(np.abs(temp_actual - ensemble_temp_pred))

print(f"\n📊 Voltage Prediction:")
print(f"  MoE MAPE:      {moe_voltage_mape:.4f}%")
print(f"  Ensemble MAPE: {ensemble_voltage_mape:.4f}%")
print(f"  Winner: {'⭐ MoE' if moe_voltage_mape < ensemble_voltage_mape else '⭐ Ensemble'}")

print(f"\n🌡️ Temperature Prediction:")
print(f"  MoE MAE:      {moe_temp_mae:.4f} °C")
print(f"  Ensemble MAE: {ensemble_temp_mae:.4f} °C")
print(f"  Winner: {'⭐ MoE' if moe_temp_mae < ensemble_temp_mae else '⭐ Ensemble'}")

# Create visualization
print("\n📈 Creating comparison plots...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

time_steps = np.arange(len(voltage_actual))

# Voltage plot
ax1.plot(time_steps, voltage_actual, 'k-', linewidth=2, label='Actual', marker='o', markersize=3)
ax1.plot(time_steps, moe_voltage_pred, 'g--', linewidth=1.5, label=f'⭐ MoE (MAPE: {moe_voltage_mape:.3f}%)', alpha=0.8)
ax1.plot(time_steps, ensemble_voltage_pred, 'b--', linewidth=1.5, label=f'Ensemble (MAPE: {ensemble_voltage_mape:.3f}%)', alpha=0.8)
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Voltage (V)', fontsize=12)
ax1.set_title(f'Voltage Prediction vs Actual - KIT Dataset Segment (Index {START_INDEX})', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Temperature plot
ax2.plot(time_steps, temp_actual, 'k-', linewidth=2, label='Actual', marker='o', markersize=3)
ax2.plot(time_steps, moe_temp_pred, 'g--', linewidth=1.5, label=f'⭐ MoE (MAE: {moe_temp_mae:.3f}°C)', alpha=0.8)
ax2.plot(time_steps, ensemble_temp_pred, 'b--', linewidth=1.5, label=f'Ensemble (MAE: {ensemble_temp_mae:.3f}°C)', alpha=0.8)
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Temperature (°C)', fontsize=12)
ax2.set_title(f'Temperature Prediction vs Actual - KIT Dataset Segment (Index {START_INDEX})', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'segment_comparison_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {output_path}")

# Save numerical results
results_dict = {
    'segment_start_index': START_INDEX,
    'sequence_length': SEQUENCE_LENGTH,
    'initial_conditions': {
        'voltage': initial_voltage,
        'temperature': initial_temp,
        'soh': soh,
        'relative_age': relative_age
    },
    'moe_metrics': {
        'voltage_mape': float(moe_voltage_mape),
        'temperature_mae': float(moe_temp_mae)
    },
    'ensemble_metrics': {
        'voltage_mape': float(ensemble_voltage_mape),
        'temperature_mae': float(ensemble_temp_mae)
    }
}

import json
with open('segment_comparison_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"✓ Results saved to: segment_comparison_results.json")

print("\n✅ Comparison complete! Check the output files.")
