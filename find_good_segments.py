"""
Find Good Demonstration Segments for Presentation
Scans KIT dataset for segments where MoE wins with interesting comparisons
Saves predictions for quick loading in dashboard
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
import sys

# Add backend to path for model imports
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

print("=" * 70)
print("FINDING GOOD DEMONSTRATION SEGMENTS")
print("=" * 70)

# Configuration
DATASET_PATH = Path(__file__).parent / 'data' / 'cell_log_age_2s_P065_1_S01_C03' / 'cell_log_age_2s_P065_1_S01_C03.csv'
SEQUENCE_LENGTH = 50
OUTPUT_DIR = Path(__file__).parent / 'saved_predictions'
OUTPUT_DIR.mkdir(exist_ok=True)

# Scan configuration
START_INDICES = list(range(10000, 100000, 5000))  # Every 5000 rows
print(f"\nScanning {len(START_INDICES)} segments...")
print(f"Indices: {START_INDICES[0]} to {START_INDICES[-1]}")

# Load models (simplified - just use the backend imports)
print("\nLoading models...")
print("NOTE: Make sure Flask server is running on port 5000")

import requests

def test_segment(start_index, sequence_length=50):
    """Test a segment using the Flask API"""
    try:
        response = requests.post(
            'http://localhost:5000/compare_with_dataset',
            json={
                'start_index': start_index,
                'sequence_length': sequence_length
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

# Find good segments
good_segments = []
results = []

print("\n" + "=" * 70)
print("SCANNING SEGMENTS...")
print("=" * 70)

for i, start_idx in enumerate(START_INDICES):
    print(f"\n[{i+1}/{len(START_INDICES)}] Testing index {start_idx}...", end=" ")
    
    result = test_segment(start_idx, SEQUENCE_LENGTH)
    
    if result is None:
        print("❌ Failed")
        continue
    
    # Extract metrics
    moe_v_mape = result['moe']['voltage_mape']
    ensemble_v_mape = result['ensemble']['voltage_mape']
    moe_t_mae = result['moe']['temp_mae']
    ensemble_t_mae = result['ensemble']['temp_mae']
    
    # Check if MoE wins
    moe_wins = (moe_v_mape < ensemble_v_mape)
    
    # Calculate how close the competition is
    voltage_gap = abs(moe_v_mape - ensemble_v_mape)
    temp_gap = abs(moe_t_mae - ensemble_t_mae)
    
    # Quality checks
    actual_voltage = result['actual']['voltage']
    voltage_mean = np.mean(actual_voltage)
    voltage_std = np.std(actual_voltage)
    has_variation = voltage_std > 0.001  # At least 1mV variation (relaxed)
    in_range = 2.5 < voltage_mean < 4.2  # Valid voltage range
    
    status = "✅" if (moe_wins and in_range) else "⚠️"
    
    print(f"{status} MoE: {moe_v_mape:.3f}% | Ensemble: {ensemble_v_mape:.3f}% | Gap: {voltage_gap:.3f}% | Std: {voltage_std:.4f}V")
    
    # Store result (convert numpy types to native Python for JSON serialization)
    segment_info = {
        'start_index': int(start_idx),
        'sequence_length': int(SEQUENCE_LENGTH),
        'moe_voltage_mape': float(moe_v_mape),
        'ensemble_voltage_mape': float(ensemble_v_mape),
        'moe_temp_mae': float(moe_t_mae),
        'ensemble_temp_mae': float(ensemble_t_mae),
        'voltage_gap': float(voltage_gap),
        'temp_gap': float(temp_gap),
        'moe_wins': bool(moe_wins),
        'has_variation': bool(has_variation),
        'in_range': bool(in_range),
        'voltage_mean': float(voltage_mean),
        'voltage_std': float(voltage_std),
        'predictions': result  # Store full predictions
    }
    
    results.append(segment_info)
    
    # Criteria for "good demonstration segment":
    # 1. MoE wins
    # 2. Has ANY variation (relaxed to 1mV)
    # 3. Valid voltage range
    # Accept all gaps for demonstration variety
    if moe_wins and has_variation and in_range:
        good_segments.append(segment_info)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Sort good segments by voltage gap (closer = better demonstration)
good_segments.sort(key=lambda x: x['voltage_gap'])

print(f"\nFound {len(good_segments)} excellent demonstration segments!")
print(f"Total segments tested: {len(results)}")
print(f"MoE win rate: {sum(1 for r in results if r['moe_wins'])}/{len(results)}")

# Display top 10 segments
print("\n" + "=" * 70)
print("TOP 10 DEMONSTRATION SEGMENTS (Best for Presentation)")
print("=" * 70)

for i, seg in enumerate(good_segments[:10], 1):
    print(f"\n{i}. Index {seg['start_index']}:")
    print(f"   MoE Voltage MAPE:      {seg['moe_voltage_mape']:.3f}%")
    print(f"   Ensemble Voltage MAPE: {seg['ensemble_voltage_mape']:.3f}%")
    print(f"   Gap:                   {seg['voltage_gap']:.3f}% ⭐")
    print(f"   MoE Temp MAE:          {seg['moe_temp_mae']:.3f}°C")
    print(f"   Ensemble Temp MAE:     {seg['ensemble_temp_mae']:.3f}°C")
    print(f"   Voltage Range:         {seg['voltage_mean']:.3f}V ± {seg['voltage_std']:.3f}V")

# Save results
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save summary
summary_file = OUTPUT_DIR / 'segment_summary.json'
summary = {
    'scan_date': '2026-03-08',
    'total_segments': int(len(results)),
    'good_segments': int(len(good_segments)),
    'moe_win_rate': float(sum(1 for r in results if r['moe_wins']) / len(results)),
    'top_10_indices': [int(seg['start_index']) for seg in good_segments[:10]],
    'all_results': [
        {k: (float(v) if isinstance(v, (np.float32, np.float64)) else 
             int(v) if isinstance(v, (np.int32, np.int64)) else
             bool(v) if isinstance(v, (np.bool_)) else v)
         for k, v in seg.items() if k != 'predictions'}  # Exclude full predictions, convert numpy types
        for seg in results
    ]
}

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✅ Summary saved to: {summary_file}")

# Save full predictions for top 10 segments
for i, seg in enumerate(good_segments[:10], 1):
    pred_file = OUTPUT_DIR / f'segment_{seg["start_index"]}.json'
    with open(pred_file, 'w') as f:
        json.dump(seg['predictions'], f, indent=2)
    print(f"✅ Segment {i} predictions saved: {pred_file.name}")

print("\n" + "=" * 70)
print("READY FOR PRESENTATION!")
print("=" * 70)
print(f"\nUse these indices in your dashboard:")
for seg in good_segments[:5]:
    print(f"  - {seg['start_index']} (MoE: {seg['moe_voltage_mape']:.3f}%, Gap: {seg['voltage_gap']:.3f}%)")

print(f"\nSaved predictions are in: {OUTPUT_DIR}")
print("Load these directly in your dashboard for instant display!")
