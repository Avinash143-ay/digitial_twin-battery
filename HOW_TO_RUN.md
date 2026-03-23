cd "c:\Users\GUDA AVINASH REDDY\Downloads\batteries\untitled folder\backend"; python backend.py
start "http://127.0.0.1:5000"

Adaptive website flow (when you have actual values):
1) Open `Compare Mode` tab.
2) In `Adaptive Improve with Actual Data`, upload CSV with columns: `current, voltage_actual, temperature_actual`.
3) Click `Evaluate vs Actual`.
4) If threshold exceeds, click `Queue Exceeded Models` (or enable auto queue).
5) Click `Tune MoE (Adapter)` or `Tune Ensemble (Full)` and monitor `Training status`.

Optional: Start MoE adapter tuning (LoRA-style, based on queued high-error samples)
Invoke-RestMethod -Method Post -Uri "http://localhost:5000/train_queued_model" -ContentType "application/json" -Body (@{
	model_name = "moe"
	tuning_mode = "adapter"
	epochs = 2
	max_samples = 50
	learning_rate = 0.0001
	rank = 8
	alpha = 16
	dropout = 0.05
	batch_size = 4
	accumulation_steps = 2
} | ConvertTo-Json)

Check training status:
Invoke-RestMethod -Method Get -Uri "http://localhost:5000/training_status"

Quick correctness check with test CSV:
1) Use file: c:\Users\GUDA AVINASH REDDY\Downloads\batteries\untitled folder\data\adaptive_test_sample.csv
2) Start backend and open website.
3) Go to Compare Mode -> Adaptive Improve with Actual Data.
4) Upload adaptive_test_sample.csv.
5) Click Evaluate vs Actual.
6) Verify charts show Actual vs MoE vs Ensemble and summary shows MAPE/MAE values.
7) If threshold exceeded, queue and tune from the same panel.
