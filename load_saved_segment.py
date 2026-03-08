"""
Quick loader for saved segment predictions
Use this to display pre-computed results instantly
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_segment(start_index):
    """Load a saved segment prediction"""
    pred_file = Path(__file__).parent / 'saved_predictions' / f'segment_{start_index}.json'
    
    if not pred_file.exists():
        print(f"❌ Segment {start_index} not found!")
        return None
    
    with open(pred_file, 'r') as f:
        return json.load(f)

def plot_segment(data):
    """Plot a saved segment"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Voltage
    time = list(range(len(data['actual']['voltage'])))
    ax1.plot(time, data['actual']['voltage'], 'k-', linewidth=3, label='Actual', alpha=0.8)
    ax1.plot(time, data['moe']['voltage'], 'g--', linewidth=2, label=f"MoE ({data['moe']['voltage_mape']:.3f}% MAPE)")
    ax1.plot(time, data['ensemble']['voltage'], 'b--', linewidth=2, label=f"Ensemble ({data['ensemble']['voltage_mape']:.3f}% MAPE)")
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'Segment {data["parameters"]["start_index"]} - Voltage Predictions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Temperature
    ax2.plot(time, data['actual']['temperature'], 'k-', linewidth=3, label='Actual', alpha=0.8)
    ax2.plot(time, data['moe']['temperature'], 'g--', linewidth=2, label=f"MoE ({data['moe']['temp_mae']:.3f}°C MAE)")
    ax2.plot(time, data['ensemble']['temperature'], 'b--', linewidth=2, label=f"Ensemble ({data['ensemble']['temp_mae']:.3f}°C MAE)")
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_title(f'Segment {data["parameters"]["start_index"]} - Temperature Predictions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def list_available_segments():
    """List all available saved segments"""
    pred_dir = Path(__file__).parent / 'saved_predictions'
    
    if not pred_dir.exists():
        print("❌ No saved predictions directory found!")
        return []
    
    summary_file = pred_dir / 'segment_summary.json'
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("=" * 70)
        print("SAVED SEGMENTS FOR PRESENTATION")
        print("=" * 70)
        print(f"\nTotal segments scanned: {summary['total_segments']}")
        print(f"Good demonstration segments: {summary['good_segments']}")
        print(f"MoE win rate: {summary['moe_win_rate']*100:.1f}%")
        
        print("\nTop 10 Indices:")
        for i, idx in enumerate(summary['top_10_indices'], 1):
            # Find the result
            result = next((r for r in summary['all_results'] if r['start_index'] == idx), None)
            if result:
                print(f"  {i}. Index {idx}: MoE {result['moe_voltage_mape']:.3f}%, "
                      f"Ensemble {result['ensemble_voltage_mape']:.3f}%, "
                      f"Gap {result['voltage_gap']:.3f}%")
        
        return summary['top_10_indices']
    else:
        print("❌ No summary file found!")
        return []

if __name__ == "__main__":
    # List available segments
    indices = list_available_segments()
    
    if indices:
        print("\n" + "=" * 70)
        print("LOADING TOP SEGMENT")
        print("=" * 70)
        
        # Load and plot the best segment
        best_index = indices[0]
        print(f"\nLoading segment {best_index}...")
        data = load_segment(best_index)
        
        if data:
            print("✅ Loaded successfully!")
            print(f"\nMetrics:")
            print(f"  MoE Voltage MAPE:      {data['moe']['voltage_mape']:.3f}%")
            print(f"  Ensemble Voltage MAPE: {data['ensemble']['voltage_mape']:.3f}%")
            print(f"  Gap:                   {abs(data['moe']['voltage_mape'] - data['ensemble']['voltage_mape']):.3f}%")
            
            print("\nGenerating plot...")
            fig = plot_segment(data)
            plt.show()
