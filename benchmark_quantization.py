import copy
import os
import time
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, input_dim1=4, input_dim2=1, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.fc_combined = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_combined_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, output_dim)

    def input_processing(self, initial_state, action):
        x = initial_state.clone()
        x[:, 0] = x[:, 0] * 10
        x[:, 1] = x[:, 1] / 3
        x[:, 2] = x[:, 2] / 30

        a = action.clone() / 5
        return x, a

    def forward(self, initial_state, action):
        original = initial_state.clone()
        x, a = self.input_processing(initial_state, action)

        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(a))
        h = torch.cat([h1, h2], dim=1)
        h = F.gelu(self.fc_combined(h))
        h = F.gelu(self.fc_combined_2(h))
        out = self.out(h)

        final_voltage = original[:, 1] + out[:, 0]
        final_temperature = original[:, 2] + out[:, 1]
        return torch.stack([final_voltage, final_temperature], dim=1)


class DeepEnsemble(nn.Module):
    def __init__(self, num_models=10, input_dim1=4, input_dim2=1, hidden_dim=128, output_dim=2):
        super().__init__()
        self.models = nn.ModuleList([
            BaseModel(input_dim1, input_dim2, hidden_dim, output_dim)
            for _ in range(num_models)
        ])

    def forward(self, initial_state, action):
        outputs = [model(initial_state, action) for model in self.models]
        return torch.stack(outputs, dim=0)


def load_ensemble_model(weights_path):
    model = DeepEnsemble(num_models=10)
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def benchmark(model, steps=75, repeats=80):
    initial_state = torch.tensor([[0.35, 3.7, 25.0, 0.0]], dtype=torch.float32)
    action = torch.tensor([[0.15]], dtype=torch.float32)

    with torch.no_grad():
        for _ in range(10):
            model(initial_state, action)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            current_v = 3.7
            current_t = 25.0
            current_i = 0.0
            for _ in range(steps):
                initial_state[0, 1] = current_v
                initial_state[0, 2] = current_t
                initial_state[0, 3] = current_i
                preds = model(initial_state, action)
                current_v = float(torch.median(preds[:, 0, 0]))
                current_t = float(torch.median(preds[:, 0, 1]))
                current_i = float(action.item())
    end = time.perf_counter()
    return end - start


def serialized_state_size_bytes(model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        temp_path = tmp.name
    try:
        torch.save(model.state_dict(), temp_path)
        return os.path.getsize(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(repo_root, "models", "digital_twin_simpler.pt")

    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return

    fp32_model = load_ensemble_model(weights_path)
    int8_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(fp32_model),
        {nn.Linear},
        dtype=torch.qint8,
    )
    int8_model.eval()

    fp32_time = benchmark(fp32_model)
    int8_time = benchmark(int8_model)

    fp32_size = serialized_state_size_bytes(fp32_model)
    int8_size = serialized_state_size_bytes(int8_model)

    speedup = fp32_time / int8_time if int8_time > 0 else 0.0
    size_reduction = (1.0 - (int8_size / fp32_size)) * 100 if fp32_size > 0 else 0.0

    print("=== DeepEnsemble Quantization Benchmark ===")
    print(f"FP32 time: {fp32_time:.4f} s")
    print(f"INT8 time: {int8_time:.4f} s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"FP32 serialized state size: {fp32_size / (1024 * 1024):.2f} MB")
    print(f"INT8 serialized state size: {int8_size / (1024 * 1024):.2f} MB")
    print(f"Serialized state reduction: {size_reduction:.2f}%")


if __name__ == "__main__":
    main()
