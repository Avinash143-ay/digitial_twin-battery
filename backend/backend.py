from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
import threading
import time
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(script_dir, '..', 'frontend')

app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)  # Enable CORS for browser requests

KIT_BLOCK_SIZE = 500
KIT_TRAIN_BLOCK_SIZE = 300
KIT_VAL_BLOCK_SIZE = 100
KIT_TEST_BLOCK_SIZE = 100
ENSEMBLE_QUANTIZATION_ENABLED = os.environ.get('ENABLE_ENSEMBLE_QUANTIZATION', '0') == '1'
ENSEMBLE_PRECISION_MODE = os.environ.get('ENSEMBLE_PRECISION_MODE', 'fp32').strip().lower()

if ENSEMBLE_PRECISION_MODE not in ('fp32', 'fp16'):
    ENSEMBLE_PRECISION_MODE = 'fp32'

# Define the DeepEnsemble model architecture
class BaseModel(nn.Module):
    def __init__(self, input_dim1=4, input_dim2=1, hidden_dim=128, output_dim=2):
        super(BaseModel, self).__init__()
        # For the first input
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        # For the second input
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        # Combined layers
        self.fc_combined = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_combined_2 = nn.Linear(hidden_dim, hidden_dim//2)
    
        self.out = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, initial_state, action):
        original_initial_state = initial_state.clone()
        action_original = action.clone()
        batch_size = original_initial_state.shape[0]
        initial_state_processed, actions_processed = self.input_processing(initial_state, action_original)
    
        h1 = F.gelu(self.fc1(initial_state_processed))
        h2 = F.gelu(self.fc2(actions_processed))
        h = torch.cat([h1, h2], dim=1)
        h = F.gelu(self.fc_combined(h))
        h = F.gelu(self.fc_combined_2(h))
        
        out = self.out(h)

        original_voltage = original_initial_state[:,1]
        original_temperature = original_initial_state[:,2]

        final_voltage = original_voltage + out[:,0]
        final_temperature = original_temperature + out[:,1]

        final_output = torch.stack([final_voltage, final_temperature], dim=1)
        return final_output
    
    def input_processing(self, initial_state, action):
        original_initial_state = initial_state.clone()
        original_initial_state[:,0] = original_initial_state[:,0]*10
        original_initial_state[:,1] = original_initial_state[:,1]/3
        original_initial_state[:,2] = original_initial_state[:,2]/30

        original_actions = action.clone()
        original_actions = original_actions/5
       
        return original_initial_state, original_actions

class DeepEnsemble(nn.Module):
    def __init__(self, num_models=10, input_dim1=4, input_dim2=1, hidden_dim=128, output_dim=2):
        super(DeepEnsemble, self).__init__()
        self.models = nn.ModuleList([
            BaseModel(input_dim1, input_dim2, hidden_dim, output_dim) 
            for _ in range(num_models)
        ])

    def forward(self, initial_state, action):
        outputs = [model(initial_state, action) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        return outputs

# MoE Layer for the Digital Twin
class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, aggregate_expert_hparam=1000, top_k=20,
                 calculated_to_noise_ratio=1):
        super(MoELayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregate_hparam = aggregate_expert_hparam
        self.aggregate_expert_size = (self.aggregate_hparam) * (input_dim * output_dim)
        self.aggregate_expert_number = self.aggregate_expert_size // self.input_dim
        self.MoE_shape = (self.aggregate_expert_number, self.input_dim)
        self.mixture_experts = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.MoE_shape)))
        self.top_K = top_k
        self.c2nr = calculated_to_noise_ratio
        
    def forward(self, x, calculated_to_noise_ratio):
        scores = torch.einsum("ijk,lk->ijl", x, self.mixture_experts)
        scores = scores.reshape(x.shape[0], x.shape[1], self.output_dim, self.aggregate_hparam)
        score_softmax = torch.mean(scores, dim=2)
        
        softmax_output = torch.softmax(score_softmax, dim=2)
        gaussian_tensor = torch.softmax(torch.randn(score_softmax.shape, device=x.device), dim=2)
        
        orignal_ratio = self.c2nr * softmax_output
        noise = (1 - self.c2nr) * gaussian_tensor
        softmax_output_final = orignal_ratio + noise
        
        topk_indices = torch.topk(softmax_output_final, self.top_K, dim=2).indices
        topk_indices = topk_indices.unsqueeze(-1)
        scores_ = scores.permute(0, 1, 3, 2)
        scores_ = torch.gather(scores_, dim=2, index=topk_indices.expand(-1, -1, -1, self.output_dim))
        output = torch.sum(scores_, dim=2)
        return output

# MoE-Enhanced Digital Twin Model
class Digital_Twin_v1(nn.Module):
    def __init__(self, initial_state_dim, action_max_length, 
                 output_state_dim, pos_encoding_dims=3, hidden_dim=64, c2nr=1):
        super(Digital_Twin_v1, self).__init__()
        self.initial_state_dim = initial_state_dim
        self.action_max_length = action_max_length
        self.output_state_dim = output_state_dim
        self.pos_encoding_dims = pos_encoding_dims
        self.c2nr = c2nr
        self.positional_encoding = self.initialize_positional_encoding(pos_encoding_dims=self.pos_encoding_dims)

        self.embedding_layer = nn.Linear(self.pos_encoding_dims + initial_state_dim + 2, hidden_dim)

        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=64, nhead=2,
                                                     batch_first=True, dim_feedforward=256, dropout=0)
        self.encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=2)

        self.Moe_layer_1 = MoELayer(input_dim=64, output_dim=100, top_k=20, aggregate_expert_hparam=100)
        self.Moe_layer_12 = MoELayer(input_dim=100, output_dim=50, top_k=20, aggregate_expert_hparam=100)
        self.Moe_layer_13 = MoELayer(input_dim=10, output_dim=2, top_k=20, aggregate_expert_hparam=100)
        self.linear_final = nn.Linear(10, 2)

        self.layer_norm1 = nn.LayerNorm(100)
        self.layer_norm12 = nn.LayerNorm(50)
        self.dropout_p = 0.15
        self.dropout_object = nn.Dropout(p=self.dropout_p)

        self.volt_lin_final1 = nn.Linear(50, 30)
        self.volt_lin_final2 = nn.Linear(30, 30)

        self.volt_lin_esti_1_mu = nn.Linear(30, 30)
        self.volt_lin_esti_1_mu_trans = nn.TransformerEncoderLayer(d_model=30, nhead=2, batch_first=True, dim_feedforward=128)
        self.volt_lin_esti_2_mu = nn.Linear(30, 10)
        self.volt_lin_esti_2_mu_trans = nn.TransformerEncoderLayer(d_model=10, nhead=2, batch_first=True, dim_feedforward=32)
        self.volt_lin_esti_3_mu = nn.Linear(10, 1)

        self.volt_lin_esti_1_sigma = nn.Linear(30, 30)
        self.volt_lin_esti_1_sigma_trans = nn.TransformerEncoderLayer(d_model=30, nhead=2, batch_first=True, dim_feedforward=128)
        self.volt_lin_esti_2_sigma = nn.Linear(30, 10)
        self.volt_lin_esti_2_sigma_trans = nn.TransformerEncoderLayer(d_model=10, nhead=2, batch_first=True, dim_feedforward=32)
        self.volt_lin_esti_3_sigma = nn.Linear(10, 1)

        self.temp_lin_final1 = nn.Linear(50, 30)
        self.temp_lin_final2 = nn.Linear(30, 30)
        
        self.temp_lin_esti_1_mu = nn.Linear(30, 30)
        self.temp_lin_esti_1_mu_trans = nn.TransformerEncoderLayer(d_model=30, nhead=2, batch_first=True, dim_feedforward=128)
        self.temp_lin_esti_2_mu = nn.Linear(30, 10)
        self.temp_lin_esti_2_mu_trans = nn.TransformerEncoderLayer(d_model=10, nhead=2, batch_first=True, dim_feedforward=32)
        self.temp_lin_esti_3_mu = nn.Linear(10, 1)

        self.temp_lin_esti_1_sigma = nn.Linear(30, 30)
        self.temp_lin_esti_1_sigma_trans = nn.TransformerEncoderLayer(d_model=30, nhead=2, batch_first=True, dim_feedforward=128)
        self.temp_lin_esti_2_sigma = nn.Linear(30, 10)
        self.temp_lin_esti_2_sigma_trans = nn.TransformerEncoderLayer(d_model=10, nhead=2, batch_first=True, dim_feedforward=32)
        self.temp_lin_esti_3_sigma = nn.Linear(10, 1)

    def initialize_positional_encoding(self, pos_encoding_dims=4):
        positional_encoding = torch.arange(self.action_max_length)
        positional_encoding = positional_encoding.unsqueeze(1).repeat(1, pos_encoding_dims)

        denominator_exponent_array = torch.arange(pos_encoding_dims)
        denominator_exponent_array = (2 * (denominator_exponent_array // 2) / pos_encoding_dims)
        denominator = torch.pow(self.action_max_length, denominator_exponent_array)
        inputs_before_function = positional_encoding / denominator
        positional_encoding_ = torch.full_like(inputs_before_function, -1)
        positional_encoding_[:, 0::2] = torch.sin(inputs_before_function[:, 0::2])
        positional_encoding_[:, 1::2] = torch.cos(inputs_before_function[:, 1::2])
        
        return positional_encoding_
    
    def input_processing(self, initial_state, action):
        original_initial_state = initial_state.clone()
        original_initial_state[:, 0] = original_initial_state[:, 0] * 10  # Normalization for relative_age
        original_initial_state[:, 1] = original_initial_state[:, 1] / 3  # Voltage Normalization
        original_initial_state[:, 2] = original_initial_state[:, 2] / 30  # Temperature Normalization
        original_actions = action.clone()
        original_actions = original_actions / 5  # Current Normalization
        
        actions_clone_1 = action.clone()
        actions_clone_2 = action.clone()
        actions_delta_shifted = actions_clone_2
        actions_delta_shifted[:, 1:, :] -= actions_clone_1[:, 0:-1, :]  # Capture increase every t
        actions_delta_shifted[:, 0, :] = 0

        modified_initial_state = original_initial_state.unsqueeze(1).repeat(1, original_actions.shape[1], 1)
        concatenated_features = torch.cat((modified_initial_state, actions_delta_shifted, original_actions), dim=2)

        return concatenated_features

    def forward(self, initial_state, action, inference_mode=False):
        original_initial_state = initial_state.clone()
        batch_size = original_initial_state.shape[0]
        sequence_length = action.size(1)
        
        concatenated_features = self.input_processing(initial_state, action)
        # Use only the positional encoding for the actual sequence length
        positional_encoding_transformed = self.positional_encoding[:sequence_length, :].unsqueeze(0).repeat(concatenated_features.shape[0], 1, 1)
        position_encoded_features = torch.cat((concatenated_features, positional_encoding_transformed), dim=2)
        
        embedding = self.embedding_layer(position_encoded_features)
        residual_1 = self.encoder_1(embedding)
        residual_1 = self.Moe_layer_1(residual_1, calculated_to_noise_ratio=1)
        residual_1 = self.layer_norm1(residual_1)
        residual_1 = F.gelu(residual_1)
        residual_1 = self.Moe_layer_12(residual_1, calculated_to_noise_ratio=1)
        residual_1 = self.layer_norm12(residual_1)
        residual_1 = F.gelu(residual_1)

        residual_1_v = F.gelu(self.volt_lin_final1(residual_1))
        residual_1_v = self.volt_lin_final2(residual_1_v)

        residual_1_t = F.gelu(self.temp_lin_final1(residual_1))
        residual_1_t = self.temp_lin_final2(residual_1_t)
        
        residual_1_v_mu = F.gelu(self.volt_lin_esti_1_mu(residual_1_v))
        residual_1_v_mu = self.volt_lin_esti_1_mu_trans(residual_1_v_mu)
        residual_1_v_mu = F.gelu(self.volt_lin_esti_2_mu(residual_1_v_mu))
        residual_1_v_mu = self.volt_lin_esti_2_mu_trans(residual_1_v_mu)
        residual_1_v_mu = self.volt_lin_esti_3_mu(residual_1_v_mu)

        residual_1_v_sigma = F.gelu(self.volt_lin_esti_1_sigma(residual_1_v))
        residual_1_v_sigma = self.volt_lin_esti_1_sigma_trans(residual_1_v_sigma)
        residual_1_v_sigma = F.gelu(self.volt_lin_esti_2_sigma(residual_1_v_sigma))
        residual_1_v_sigma = self.volt_lin_esti_2_sigma_trans(residual_1_v_sigma)
        residual_1_v_sigma = self.volt_lin_esti_3_sigma(residual_1_v_sigma)

        residual_1_t_mu = F.gelu(self.temp_lin_esti_1_mu(residual_1_t))
        residual_1_t_mu = self.temp_lin_esti_1_mu_trans(residual_1_t_mu)
        residual_1_t_mu = F.gelu(self.temp_lin_esti_2_mu(residual_1_t_mu))
        residual_1_t_mu = self.temp_lin_esti_2_mu_trans(residual_1_t_mu)
        residual_1_t_mu = self.temp_lin_esti_3_mu(residual_1_t_mu)

        residual_1_t_sigma = F.gelu(self.temp_lin_esti_1_sigma(residual_1_t))
        residual_1_t_sigma = self.temp_lin_esti_1_sigma_trans(residual_1_t_sigma)
        residual_1_t_sigma = F.gelu(self.temp_lin_esti_2_sigma(residual_1_t_sigma))
        residual_1_t_sigma = self.temp_lin_esti_2_sigma_trans(residual_1_t_sigma)
        residual_1_t_sigma = self.temp_lin_esti_3_sigma(residual_1_t_sigma)

        voltage_mu_prediction = residual_1_v_mu.squeeze(dim=-1)  # [batch, seq, 1] -> [batch, seq]
        voltage_error_prediction = torch.exp(residual_1_v_sigma.squeeze(dim=-1))
        voltage_error_prediction = torch.clamp(voltage_error_prediction, min=0, max=50)

        temp_mu_prediction = residual_1_t_mu.squeeze(dim=-1)
        temp_error_prediction = torch.exp(residual_1_t_sigma.squeeze(dim=-1))
        temp_error_prediction = torch.clamp(temp_error_prediction, min=0, max=50)
        
        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, action.size(1), 1)
        repeated_original_voltage = repeated_original_initial_state[:, :, 1]
        repeated_original_temperature = repeated_original_initial_state[:, :, 2]

        predicted_states_voltage = repeated_original_voltage + voltage_mu_prediction
        predicted_states_temperature = repeated_original_temperature + temp_mu_prediction
        
        predicted_mu_sigma_final = torch.stack((predicted_states_voltage, predicted_states_temperature,
                                                voltage_error_prediction, temp_error_prediction), dim=2)
        return predicted_mu_sigma_final


class AdapterLinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))

        self.adapter_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.adapter_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        self.merged = False

        nn.init.kaiming_uniform_(self.adapter_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.adapter_B.weight)

    def _delta_weight(self):
        # Equivalent merged adapter update for Linear: W + scale * (B @ A)
        return torch.matmul(self.adapter_B.weight, self.adapter_A.weight)

    def merge_adapter(self):
        if self.merged:
            return False
        delta = self._delta_weight().to(self.base_layer.weight.dtype)
        with torch.no_grad():
            self.base_layer.weight.add_(self.scale * delta)
        self.merged = True
        return True

    def unmerge_adapter(self):
        if not self.merged:
            return False
        delta = self._delta_weight().to(self.base_layer.weight.dtype)
        with torch.no_grad():
            self.base_layer.weight.sub_(self.scale * delta)
        self.merged = False
        return True

    def forward(self, x):
        base_out = self.base_layer(x)
        if self.merged:
            return base_out
        delta = self.adapter_B(self.dropout(self.adapter_A(x)))
        return base_out + self.scale * delta


DEFAULT_MOE_ADAPTER_TARGETS = [
    'embedding_layer',
    'volt_lin_final1', 'volt_lin_final2',
    'temp_lin_final1', 'temp_lin_final2',
    'volt_lin_esti_1_mu', 'volt_lin_esti_2_mu', 'volt_lin_esti_3_mu',
    'volt_lin_esti_1_sigma', 'volt_lin_esti_2_sigma', 'volt_lin_esti_3_sigma',
    'temp_lin_esti_1_mu', 'temp_lin_esti_2_mu', 'temp_lin_esti_3_mu',
    'temp_lin_esti_1_sigma', 'temp_lin_esti_2_sigma', 'temp_lin_esti_3_sigma'
]


def get_module_by_path(root_module, module_path):
    module = root_module
    for part in module_path.split('.'):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_path(root_module, module_path, new_module):
    parts = module_path.split('.')
    parent = root_module
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last_part = parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)


def ensure_moe_adapters(base_model, target_modules, rank=8, alpha=16.0, dropout=0.05):
    applied_modules = []
    for module_name in target_modules:
        try:
            original_module = get_module_by_path(base_model, module_name)
        except Exception:
            continue

        if isinstance(original_module, AdapterLinear):
            continue

        if isinstance(original_module, nn.Linear):
            adapter_layer = AdapterLinear(original_module, rank=rank, alpha=alpha, dropout=dropout)
            set_module_by_path(base_model, module_name, adapter_layer)
            applied_modules.append(module_name)

    return applied_modules


def get_moe_adapter_parameters(base_model):
    adapter_params = []
    for module in base_model.modules():
        if isinstance(module, AdapterLinear):
            adapter_params.extend(list(module.adapter_A.parameters()))
            adapter_params.extend(list(module.adapter_B.parameters()))
    return adapter_params


def iter_moe_adapter_modules(base_model):
    for module in base_model.modules():
        if isinstance(module, AdapterLinear):
            yield module


def count_moe_adapter_modules(base_model):
    return sum(1 for _ in iter_moe_adapter_modules(base_model))


def are_moe_adapters_merged(base_model):
    adapter_modules = list(iter_moe_adapter_modules(base_model))
    if not adapter_modules:
        return False
    return all(module.merged for module in adapter_modules)


def merge_moe_adapters(base_model):
    merged_count = 0
    for module in iter_moe_adapter_modules(base_model):
        if module.merge_adapter():
            merged_count += 1
    return merged_count


def unmerge_moe_adapters(base_model):
    unmerged_count = 0
    for module in iter_moe_adapter_modules(base_model):
        if module.unmerge_adapter():
            unmerged_count += 1
    return unmerged_count


def save_moe_adapter_checkpoint(base_model, checkpoint_path, config):
    adapter_state_dict = {
        name: tensor.cpu()
        for name, tensor in base_model.state_dict().items()
        if '.adapter_A.' in name or '.adapter_B.' in name
    }

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'adapter_state_dict': adapter_state_dict,
        'config': config,
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    }, checkpoint_path)


def load_moe_adapter_checkpoint(base_model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint.get('config', {})
    target_modules = config.get('target_modules', DEFAULT_MOE_ADAPTER_TARGETS)
    rank = int(config.get('rank', 8))
    alpha = float(config.get('alpha', 16.0))
    dropout = float(config.get('dropout', 0.05))

    ensure_moe_adapters(base_model, target_modules, rank=rank, alpha=alpha, dropout=dropout)
    adapter_state_dict = checkpoint.get('adapter_state_dict', {})
    base_model.load_state_dict(adapter_state_dict, strict=False)
    return config

# Initialize and load the MoE-Enhanced Transformer model
print("Loading MoE-Enhanced Digital Twin model...")
model = Digital_Twin_v1(
    initial_state_dim=3,  # [relative_age, voltage, temperature]
    action_max_length=150,
    output_state_dim=2,  # [voltage, temperature]
    pos_encoding_dims=3,
    hidden_dim=64,
    c2nr=1
)

# Load MoE Transformer model weights
MOE_ADAPTER_PATH = os.path.join(script_dir, '..', 'models', 'moe_lora_adapter.pt')

moe_adapter_config = {
    'rank': 8,
    'alpha': 16.0,
    'dropout': 0.05,
    'target_modules': list(DEFAULT_MOE_ADAPTER_TARGETS)
}

try:
    saved_model_path = os.path.join(script_dir, '..', 'Digital_Twin', 'digital_twin_best.pt')
    checkpoint = torch.load(saved_model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("MoE-Enhanced Transformer loaded successfully from Digital_Twin/digital_twin_best.pt!")
except Exception as e:
    print(f"Warning: Could not load MoE Transformer model weights: {e}")
    print("Using untrained model for demonstration purposes.")
    model.eval()

try:
    loaded_adapter_config = load_moe_adapter_checkpoint(model, MOE_ADAPTER_PATH)
    if loaded_adapter_config:
        moe_adapter_config.update(loaded_adapter_config)
        print(f"Loaded MoE LoRA adapters from {MOE_ADAPTER_PATH}")
except Exception as e:
    print(f"Warning: Could not load MoE LoRA adapters: {e}")

# Initialize and load the DeepEnsemble model
print("Loading DeepEnsemble model...")
ensemble_model = DeepEnsemble(num_models=10)
ensemble_inference_model = ensemble_model
ensemble_inference_mode = 'fp32'
ensemble_quantization_lock = threading.Lock()


def build_quantized_ensemble_model(base_model):
    quantized_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(base_model),
        {nn.Linear},
        dtype=torch.qint8
    )
    quantized_model.eval()
    return quantized_model


def build_fp16_ensemble_model(base_model):
    fp16_model = copy.deepcopy(base_model)
    fp16_model.half()
    fp16_model.eval()
    return fp16_model


def refresh_ensemble_inference_model():
    global ensemble_inference_model, ensemble_inference_mode

    if ENSEMBLE_QUANTIZATION_ENABLED:
        try:
            ensemble_inference_model = build_quantized_ensemble_model(ensemble_model)
            ensemble_inference_mode = 'int8_dynamic'
            print('DeepEnsemble inference quantization enabled (dynamic int8).')
            return
        except Exception as e:
            print(f"Warning: Could not quantize DeepEnsemble for inference: {e}")

    if ENSEMBLE_PRECISION_MODE == 'fp16':
        try:
            ensemble_inference_model = build_fp16_ensemble_model(ensemble_model)
            ensemble_inference_mode = 'fp16'
            print('DeepEnsemble inference precision set to fp16.')
            return
        except Exception as e:
            print(f"Warning: Could not switch DeepEnsemble to fp16 for inference: {e}")

    ensemble_inference_model = ensemble_model
    ensemble_inference_mode = 'fp32'


def get_active_ensemble_model():
    return ensemble_inference_model


def get_active_ensemble_input_dtype():
    return torch.float16 if ensemble_inference_mode == 'fp16' else torch.float32

try:
    ensemble_path = os.path.join(script_dir, '..', 'models', 'digital_twin_simpler.pt')
    checkpoint = torch.load(ensemble_path, map_location=torch.device('cpu'))
    if "state_dict" in checkpoint:
        ensemble_model.load_state_dict(checkpoint["state_dict"])
    else:
        ensemble_model.load_state_dict(checkpoint)
    ensemble_model.eval()
    print("DeepEnsemble model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load DeepEnsemble model weights: {e}")
    print("Using untrained ensemble for demonstration purposes.")
    ensemble_model.eval()

refresh_ensemble_inference_model()

device = torch.device("cpu")

training_lock = threading.Lock()
moe_adapter_lock = threading.Lock()
training_state = {
    'running': False,
    'status': 'idle',
    'model_name': None,
    'tuning_mode': None,
    'message': 'No training job started yet.',
    'started_at': None,
    'finished_at': None,
    'processed_samples': 0,
    'total_samples': 0,
    'trainable_parameters': 0,
    'last_loss': None,
    'last_saved_model': None,
    'error': None
}


def update_training_state(**kwargs):
    with training_lock:
        training_state.update(kwargs)


def get_training_state_snapshot():
    with training_lock:
        return dict(training_state)


def count_dataset_rows(dataset_path):
    # Count data rows excluding header without loading the entire CSV into memory.
    total_lines = 0
    with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            total_lines += 1
    return max(0, total_lines - 1)


def build_block_split_ranges(total_rows, sequence_length):
    required_rows = max(2, int(sequence_length) + 1)
    max_start = total_rows - required_rows
    split_ranges = {
        'train': [],
        'val': [],
        'test': []
    }

    if max_start < 0:
        return split_ranges

    section_specs = (
        ('train', 0, KIT_TRAIN_BLOCK_SIZE),
        ('val', KIT_TRAIN_BLOCK_SIZE, KIT_VAL_BLOCK_SIZE),
        ('test', KIT_TRAIN_BLOCK_SIZE + KIT_VAL_BLOCK_SIZE, KIT_TEST_BLOCK_SIZE),
    )

    for block_start in range(0, total_rows, KIT_BLOCK_SIZE):
        block_end = min(block_start + KIT_BLOCK_SIZE, total_rows)

        for section_name, section_offset, section_size in section_specs:
            section_start = block_start + section_offset
            section_end = min(section_start + section_size, block_end)

            if section_start >= section_end:
                continue

            valid_end = min(section_end - required_rows, max_start)
            if section_start <= valid_end:
                split_ranges[section_name].append((section_start, valid_end))

    return split_ranges


def count_range_positions(ranges):
    return int(sum((end - start + 1) for start, end in ranges))


def index_in_ranges(index, ranges):
    return any(start <= index <= end for start, end in ranges)


def nearest_index_in_ranges(index, ranges):
    if not ranges:
        return None

    best_index = None
    best_distance = None

    for start, end in ranges:
        if index < start:
            candidate = start
        elif index > end:
            candidate = end
        else:
            return index

        distance = abs(candidate - index)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = candidate

    return best_index


def load_retraining_records(model_name, max_samples=50):
    queue_path = os.path.join(script_dir, '..', 'retraining_queue', f'{model_name}_high_error_segments.jsonl')
    if not os.path.exists(queue_path):
        return [], queue_path

    records = []
    with open(queue_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if max_samples > 0:
        records = records[-max_samples:]
    return records, queue_path


def build_moe_training_sample(record):
    params = record.get('parameters', {})
    actual = record.get('actual', {})
    current_profile = params.get('current_profile', [])
    voltage_actual = actual.get('voltage', [])
    temp_actual = actual.get('temperature', [])

    if not current_profile or not voltage_actual or not temp_actual:
        return None

    steps = min(len(current_profile), len(voltage_actual), len(temp_actual), 150)
    if steps < 2:
        return None

    relative_age_value = float(params.get('relative_age', params.get('soh', 0.65)))
    initial_voltage = float(params.get('initial_voltage', voltage_actual[0]))
    initial_temp = float(params.get('initial_temperature', temp_actual[0]))

    initial_state = torch.tensor([[relative_age_value, initial_voltage, initial_temp]], dtype=torch.float32, device=device)
    actions = torch.tensor([current_profile[:steps]], dtype=torch.float32, device=device).unsqueeze(-1)
    target_voltage = torch.tensor(voltage_actual[:steps], dtype=torch.float32, device=device)
    target_temp = torch.tensor(temp_actual[:steps], dtype=torch.float32, device=device)
    return initial_state, actions, target_voltage, target_temp


def build_ensemble_training_sample(record):
    params = record.get('parameters', {})
    actual = record.get('actual', {})
    current_profile = params.get('current_profile', [])
    voltage_actual = actual.get('voltage', [])
    temp_actual = actual.get('temperature', [])

    if not current_profile or not voltage_actual or not temp_actual:
        return None

    steps = min(len(current_profile), len(voltage_actual), len(temp_actual), 75)
    if steps < 2:
        return None

    relative_age = float(params.get('relative_age', 1 - float(params.get('soh', 0.65))))
    initial_voltage = float(params.get('initial_voltage', voltage_actual[0]))
    initial_temp = float(params.get('initial_temperature', temp_actual[0]))

    return {
        'relative_age': relative_age,
        'initial_voltage': initial_voltage,
        'initial_temperature': initial_temp,
        'current_profile': [float(v) for v in current_profile[:steps]],
        'voltage_actual': [float(v) for v in voltage_actual[:steps]],
        'temp_actual': [float(v) for v in temp_actual[:steps]],
        'steps': steps
    }


def train_moe_on_records(records, epochs=1, lr=1e-4):
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    valid_samples = 0
    last_loss = None
    for epoch in range(epochs):
        for rec in records:
            sample = build_moe_training_sample(rec)
            if sample is None:
                continue

            initial_state, actions, target_voltage, target_temp = sample

            optimizer.zero_grad()
            predictions = model(initial_state, actions)
            pred_voltage = predictions[0, :, 0]
            pred_temp = predictions[0, :, 1]

            loss_voltage = F.mse_loss(pred_voltage, target_voltage)
            loss_temp = F.mse_loss(pred_temp, target_temp)
            loss = loss_voltage + loss_temp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            valid_samples += 1
            last_loss = float(loss.item())

            update_training_state(
                processed_samples=valid_samples,
                last_loss=last_loss,
                message=f'MoE full tuning in progress: sample {valid_samples}'
            )

    model.eval()
    return valid_samples, last_loss


def train_moe_adapter_on_records(records, epochs=1, lr=1e-4, rank=8, alpha=16.0,
                                 dropout=0.05, batch_size=4, accumulation_steps=1,
                                 target_modules=None):
    if target_modules is None or not target_modules:
        target_modules = list(DEFAULT_MOE_ADAPTER_TARGETS)

    ensure_moe_adapters(model, target_modules, rank=rank, alpha=alpha, dropout=dropout)

    for param in model.parameters():
        param.requires_grad = False

    adapter_params = get_moe_adapter_parameters(model)
    if not adapter_params:
        return 0, None, 0

    for param in adapter_params:
        param.requires_grad = True

    optimizer = torch.optim.Adam(adapter_params, lr=lr)
    model.train()

    valid_samples = 0
    last_loss = None
    pending_samples = 0
    samples_per_update = max(1, int(batch_size)) * max(1, int(accumulation_steps))
    optimizer.zero_grad()

    for epoch in range(epochs):
        for rec in records:
            sample = build_moe_training_sample(rec)
            if sample is None:
                continue

            initial_state, actions, target_voltage, target_temp = sample
            predictions = model(initial_state, actions)
            pred_voltage = predictions[0, :, 0]
            pred_temp = predictions[0, :, 1]

            loss_voltage = F.mse_loss(pred_voltage, target_voltage)
            loss_temp = F.mse_loss(pred_temp, target_temp)
            raw_loss = loss_voltage + loss_temp
            loss = raw_loss / samples_per_update
            loss.backward()

            valid_samples += 1
            pending_samples += 1
            last_loss = float(raw_loss.item())

            if pending_samples >= samples_per_update:
                torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                pending_samples = 0

            update_training_state(
                processed_samples=valid_samples,
                last_loss=last_loss,
                message=f'MoE adapter tuning in progress: sample {valid_samples}'
            )

    if pending_samples > 0:
        torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    return valid_samples, last_loss, len(adapter_params)


def train_ensemble_on_records(records, epochs=1, lr=1e-4):
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=lr)
    ensemble_model.train()

    valid_samples = 0
    last_loss = None

    for epoch in range(epochs):
        epoch_records = list(records)
        random.shuffle(epoch_records)

        # Start with stronger teacher forcing and relax it across epochs.
        teacher_forcing_ratio = max(0.25, 0.8 - (0.15 * epoch))

        for rec in epoch_records:
            sample = build_ensemble_training_sample(rec)
            if sample is None:
                continue

            optimizer.zero_grad()
            total_loss = 0.0
            age = sample['relative_age']
            current_profile = sample['current_profile']
            voltage_actual = sample['voltage_actual']
            temp_actual = sample['temp_actual']
            steps = sample['steps']

            prev_voltage = sample['initial_voltage']
            prev_temp = sample['initial_temperature']
            prev_current = 0.0

            for step in range(steps):
                next_current = current_profile[step]
                initial_state = torch.tensor([[age, prev_voltage, prev_temp, prev_current]], dtype=torch.float32, device=device)
                action = torch.tensor([[next_current]], dtype=torch.float32, device=device)

                outputs = ensemble_model(initial_state, action)
                target = torch.tensor([voltage_actual[step], temp_actual[step]], dtype=torch.float32, device=device)
                target = target.unsqueeze(0).repeat(outputs.shape[0], 1)
                step_loss = F.mse_loss(outputs[:, 0, :], target)
                total_loss = total_loss + step_loss

                # Blend actual and predicted state to reduce train/inference mismatch.
                median_pred_voltage = torch.median(outputs[:, 0, 0]).detach()
                median_pred_temp = torch.median(outputs[:, 0, 1]).detach()
                median_pred_voltage = torch.clamp(median_pred_voltage, min=2.4, max=4.2)

                prev_voltage = (teacher_forcing_ratio * voltage_actual[step]) + ((1.0 - teacher_forcing_ratio) * float(median_pred_voltage.item()))
                prev_temp = (teacher_forcing_ratio * temp_actual[step]) + ((1.0 - teacher_forcing_ratio) * float(median_pred_temp.item()))
                prev_current = next_current

            total_loss = total_loss / steps
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
            optimizer.step()

            valid_samples += 1
            last_loss = float(total_loss.item())

            update_training_state(
                processed_samples=valid_samples,
                last_loss=last_loss,
                message=f'Ensemble training in progress: sample {valid_samples}'
            )

    ensemble_model.eval()
    return valid_samples, last_loss


def training_worker(model_name, epochs=1, max_samples=50, lr=1e-4, tuning_mode='full',
                    rank=8, alpha=16.0, dropout=0.05, batch_size=4,
                    accumulation_steps=1, target_modules=None):
    try:
        update_training_state(
            running=True,
            status='running',
            model_name=model_name,
            tuning_mode=tuning_mode,
            message=f'Starting {model_name.upper()} retraining ({tuning_mode})...',
            started_at=datetime.utcnow().isoformat() + 'Z',
            finished_at=None,
            processed_samples=0,
            total_samples=0,
            trainable_parameters=0,
            last_loss=None,
            error=None
        )

        records, queue_path = load_retraining_records(model_name, max_samples=max_samples)
        total_samples = len(records) * max(1, int(epochs))
        update_training_state(total_samples=total_samples, message=f'Loaded {len(records)} queued samples from {queue_path}')

        if not records:
            update_training_state(
                running=False,
                status='idle',
                message=f'No queued samples found for {model_name.upper()} at {queue_path}.',
                finished_at=datetime.utcnow().isoformat() + 'Z'
            )
            return

        if model_name == 'moe':
            if tuning_mode == 'adapter':
                restore_merged_after_training = False
                with moe_adapter_lock:
                    if are_moe_adapters_merged(model):
                        unmerge_moe_adapters(model)
                        restore_merged_after_training = True

                trained_samples, last_loss, trainable_params = train_moe_adapter_on_records(
                    records,
                    epochs=epochs,
                    lr=lr,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    batch_size=batch_size,
                    accumulation_steps=accumulation_steps,
                    target_modules=target_modules
                )
                moe_adapter_config.update({
                    'rank': int(rank),
                    'alpha': float(alpha),
                    'dropout': float(dropout),
                    'target_modules': list(target_modules or DEFAULT_MOE_ADAPTER_TARGETS)
                })
                save_path = MOE_ADAPTER_PATH
                save_moe_adapter_checkpoint(model, save_path, moe_adapter_config)
                update_training_state(trainable_parameters=trainable_params)

                if restore_merged_after_training:
                    with moe_adapter_lock:
                        merge_moe_adapters(model)
            else:
                trained_samples, last_loss = train_moe_on_records(records, epochs=epochs, lr=lr)
                save_path = os.path.join(script_dir, '..', 'models', 'moe_finetuned.pt')
                torch.save({'state_dict': model.state_dict(), 'updated_at': datetime.utcnow().isoformat() + 'Z'}, save_path)
        else:
            if tuning_mode == 'adapter':
                update_training_state(
                    running=False,
                    status='failed',
                    message='Adapter tuning is currently supported only for the MoE model.',
                    error='Unsupported tuning_mode=adapter for ensemble',
                    finished_at=datetime.utcnow().isoformat() + 'Z'
                )
                return
            trained_samples, last_loss = train_ensemble_on_records(records, epochs=epochs, lr=lr)
            save_path = os.path.join(script_dir, '..', 'models', 'ensemble_finetuned.pt')
            torch.save({'state_dict': ensemble_model.state_dict(), 'updated_at': datetime.utcnow().isoformat() + 'Z'}, save_path)
            refresh_ensemble_inference_model()

        if trained_samples == 0:
            update_training_state(
                running=False,
                status='failed',
                message=f'No valid samples could be built for {model_name.upper()} training.',
                error='Queued samples missing required fields such as parameters.current_profile or actual targets.',
                finished_at=datetime.utcnow().isoformat() + 'Z'
            )
            return

        update_training_state(
            running=False,
            status='completed',
            message=f'{model_name.upper()} retraining completed on {trained_samples} samples.',
            finished_at=datetime.utcnow().isoformat() + 'Z',
            last_loss=last_loss,
            last_saved_model=save_path
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_training_state(
            running=False,
            status='failed',
            error=str(e),
            message=f'{model_name.upper()} retraining failed.',
            finished_at=datetime.utcnow().isoformat() + 'Z'
        )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract parameters
        soh = float(data['soh'])  # Relative aging (0-1)
        voltage = float(data['voltage'])  # Initial voltage (3.2-4.2)
        temperature = float(data['temperature'])  # Temperature (°C)
        
        # Get current data array from uploaded CSV
        current_data = data.get('current_data', [])
        if not current_data:
            # Fallback to single current value if provided
            current = float(data.get('current', 0))
            steps = int(data['steps'])
            current_data = [current] * steps
        
        steps = len(current_data)
        
        # Ensure steps doesn't exceed model's max length
        if steps > 150:
            current_data = current_data[:150]
            steps = 150
        
        # Prepare initial state tensor: [batch_size, 3] = [SOH, Voltage, Temperature]
        initial_state = torch.tensor([[soh, voltage, temperature]], dtype=torch.float32)
        
        # Prepare actions: [batch_size, steps, 1] - current values from CSV
        actions = torch.tensor([current_data], dtype=torch.float32).unsqueeze(-1)
        
        # Run model inference
        with torch.no_grad():
            predictions = model(initial_state, actions)
        
        # Extract predictions: shape is [batch_size, steps, 2]
        voltage_forecast = predictions[0, :, 0].cpu().numpy().tolist()
        temperature_forecast = predictions[0, :, 1].cpu().numpy().tolist()
        
        return jsonify({
            'status': 'success',
            'voltage_forecast': voltage_forecast,
            'temperature_forecast': temperature_forecast,
            'parameters': {
                'soh': soh,
                'initial_voltage': voltage,
                'initial_temperature': temperature,
                'steps': steps
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/predict_ensemble', methods=['POST'])
def predict_ensemble():
    try:
        data = request.json
        
        # Extract parameters
        relative_age = float(data.get('relative_age', 0.5))  # 0-1
        voltage = float(data['voltage'])  # Initial voltage
        temperature = float(data['temperature'])  # Temperature (°C)
        
        # Get current data array
        current_data = data.get('current_data', [])
        if not current_data:
            current = float(data.get('current', 0))
            steps = int(data['steps'])
            current_data = [current] * steps
        
        steps = len(current_data)
        if steps > 75:
            current_data = current_data[:75]
            steps = 75
        
        active_ensemble_model = get_active_ensemble_model()

        # Initialize predictions arrays
        voltage_predictions_all = [[] for _ in range(10)]  # 10 ensemble models
        temp_predictions_all = [[] for _ in range(10)]
        
        # Current state
        current_voltage = voltage
        current_temp = temperature
        current_age = relative_age
        current_i = 0.0  # Initial current at t=0
        
        # Autoregressive prediction
        for step in range(steps):
            next_current = current_data[step]  # Current at next timestep

            input_dtype = get_active_ensemble_input_dtype()
            
            # Prepare initial state: [relative_age, voltage, temp, current_at_t0]
            initial_state = torch.tensor([[current_age, current_voltage, current_temp, current_i]], 
                                        dtype=input_dtype)
            # Action is the next current (no negation - was wrong)
            action = torch.tensor([[next_current]], dtype=input_dtype)
            
            # Get ensemble predictions
            with torch.no_grad():
                predictions = active_ensemble_model(initial_state, action)  # [num_models, batch, 2]
            
            # Store predictions from each model with voltage constraints
            for model_idx in range(10):
                pred_voltage = predictions[model_idx, 0, 0].item()
                pred_temp = predictions[model_idx, 0, 1].item()
                # Constrain voltage to realistic battery range: 2.4V - 4.2V
                pred_voltage = np.clip(pred_voltage, 2.4, 4.2)
                voltage_predictions_all[model_idx].append(pred_voltage)
                temp_predictions_all[model_idx].append(pred_temp)
            
            # Update current state with median prediction for next step
            median_voltage = np.median([predictions[i, 0, 0].item() for i in range(10)])
            median_temp = np.median([predictions[i, 0, 1].item() for i in range(10)])
            # Constrain median voltage to realistic battery range
            median_voltage = np.clip(median_voltage, 2.4, 4.2)
            current_voltage = median_voltage
            current_temp = median_temp
            current_i = next_current  # Update current for next iteration
        
        # Compute median predictions across ensemble
        voltage_median = []
        temp_median = []
        for step in range(steps):
            v_median = np.median([voltage_predictions_all[m][step] for m in range(10)])
            t_median = np.median([temp_predictions_all[m][step] for m in range(10)])
            # Ensure voltage is within battery range (extra safety check)
            v_median = np.clip(v_median, 2.4, 4.2)
            voltage_median.append(v_median)
            temp_median.append(t_median)
        
        return jsonify({
            'status': 'success',
            'voltage_forecast': voltage_median,
            'temperature_forecast': temp_median,
            'voltage_ensemble': voltage_predictions_all,
            'temperature_ensemble': temp_predictions_all,
            'inference_mode': {
                'ensemble': ensemble_inference_mode,
                'ensemble_quantization_enabled': ENSEMBLE_QUANTIZATION_ENABLED
            },
            'parameters': {
                'relative_age': relative_age,
                'initial_voltage': voltage,
                'initial_temperature': temperature,
                'steps': steps
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/dataset_split_info', methods=['POST'])
def dataset_split_info():
    try:
        data = request.json or {}
        sequence_length = int(data.get('sequence_length', 75))

        if sequence_length < 2 or sequence_length > 150:
            return jsonify({
                'status': 'error',
                'message': 'sequence_length must be between 2 and 150'
            }), 400

        dataset_path = os.path.join(script_dir, '..', 'data', 'cell_log_age_2s_P065_1_S01_C03', 'cell_log_age_2s_P065_1_S01_C03.csv')
        if not os.path.exists(dataset_path):
            return jsonify({
                'status': 'error',
                'message': f'Dataset not found at {dataset_path}'
            }), 404

        total_rows = count_dataset_rows(dataset_path)
        split_ranges = build_block_split_ranges(total_rows, sequence_length)

        train_ranges = split_ranges.get('train', [])
        val_ranges = split_ranges.get('val', [])
        test_ranges = split_ranges.get('test', [])

        if not test_ranges:
            return jsonify({
                'status': 'error',
                'message': f'No valid test segments for sequence_length={sequence_length}. Reduce sequence length.'
            }), 400

        requested_start = data.get('requested_start_index')
        try:
            requested_start_int = int(requested_start) if requested_start is not None else test_ranges[0][0]
        except Exception:
            requested_start_int = test_ranges[0][0]

        suggested_test_start = nearest_index_in_ranges(requested_start_int, test_ranges)

        return jsonify({
            'status': 'success',
            'split': {
                'strategy': f'block_{KIT_BLOCK_SIZE}_train_{KIT_TRAIN_BLOCK_SIZE}_val_{KIT_VAL_BLOCK_SIZE}_test_{KIT_TEST_BLOCK_SIZE}',
                'sequence_length': sequence_length,
                'dataset_total_rows': total_rows,
                'train_count': count_range_positions(train_ranges),
                'val_count': count_range_positions(val_ranges),
                'test_count': count_range_positions(test_ranges),
                'test_first_index': int(test_ranges[0][0]),
                'test_last_index': int(test_ranges[-1][1]),
                'suggested_test_start': int(suggested_test_start),
                'requested_start_index': int(requested_start_int)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/compare_with_dataset', methods=['POST'])
def compare_with_dataset():
    try:
        import pandas as pd
        data = request.json or {}
        
        # Get parameters
        requested_start_index = data.get('start_index', 10000)
        sequence_length = int(data.get('sequence_length', 75))
        use_test_split = bool(data.get('use_test_split', False))

        if sequence_length < 2 or sequence_length > 150:
            return jsonify({
                'status': 'error',
                'message': 'sequence_length must be between 2 and 150'
            }), 400

        try:
            requested_start_index = int(requested_start_index)
        except Exception:
            requested_start_index = 10000
        
        # Load segment from KIT dataset
        dataset_path = os.path.join(script_dir, '..', 'data', 'cell_log_age_2s_P065_1_S01_C03', 'cell_log_age_2s_P065_1_S01_C03.csv')
        
        if not os.path.exists(dataset_path):
            return jsonify({
                'status': 'error',
                'message': f'Dataset not found at {dataset_path}'
            }), 404

        total_rows = count_dataset_rows(dataset_path)
        max_start = total_rows - (sequence_length + 1)
        if max_start < 0:
            return jsonify({
                'status': 'error',
                'message': f'Not enough dataset rows for sequence_length={sequence_length}.'
            }), 400

        split_ranges = build_block_split_ranges(total_rows, sequence_length)
        test_ranges = split_ranges.get('test', [])

        if use_test_split and not test_ranges:
            return jsonify({
                'status': 'error',
                'message': f'No valid test segments for sequence_length={sequence_length}. Reduce sequence length.'
            }), 400

        start_index = requested_start_index
        if use_test_split:
            start_index = nearest_index_in_ranges(requested_start_index, test_ranges)

        if start_index is None:
            return jsonify({
                'status': 'error',
                'message': 'Could not resolve a valid start index for requested split.'
            }), 400

        if start_index < 0 or start_index > max_start:
            return jsonify({
                'status': 'error',
                'message': f'start_index must be between 0 and {max_start} for sequence_length={sequence_length}'
            }), 400

        if use_test_split and not index_in_ranges(start_index, test_ranges):
            return jsonify({
                'status': 'error',
                'message': 'Resolved start_index is not inside the test split.'
            }), 400
        
        # Read the specific segment
        df = pd.read_csv(dataset_path, sep=';', skiprows=range(1, start_index), nrows=sequence_length+1)

        if len(df) < 2:
            return jsonify({
                'status': 'error',
                'message': f'No valid segment found at start_index={start_index}. Choose a different index.'
            }), 400
        
        # Extract data
        initial_row = df.iloc[0]
        initial_voltage = float(initial_row['v_raw_V'])
        initial_temp = float(initial_row['t_cell_degC'])
        current_profile = df['i_raw_A'].values[1:sequence_length+1].tolist()
        
        # Actual values
        voltage_actual = df['v_raw_V'].values[1:sequence_length+1].tolist()
        temp_actual = df['t_cell_degC'].values[1:sequence_length+1].tolist()
        
        # Estimate SOH
        relative_age = 0.35
        soh = 1.0 - relative_age
        
        # Prepare inputs for models
        steps = len(current_profile)
        initial_state_moe = torch.tensor([[relative_age, initial_voltage, initial_temp]], dtype=torch.float32)
        actions = torch.tensor([current_profile], dtype=torch.float32).unsqueeze(-1)
        
        # Run MoE Transformer
        with torch.no_grad():
            moe_predictions = model(initial_state_moe, actions)
        moe_voltage_forecast = moe_predictions[0, :, 0].cpu().numpy().tolist()
        moe_temp_forecast = moe_predictions[0, :, 1].cpu().numpy().tolist()
        
        # Run Deep Ensemble
        active_ensemble_model = get_active_ensemble_model()
        voltage_predictions_all = [[] for _ in range(10)]
        temp_predictions_all = [[] for _ in range(10)]
        
        current_voltage = initial_voltage
        current_temp = initial_temp
        current_age = relative_age
        current_i = 0.0
        
        for step in range(steps):
            next_current = current_profile[step]
            input_dtype = get_active_ensemble_input_dtype()
            initial_state_ens = torch.tensor([[current_age, current_voltage, current_temp, current_i]], dtype=input_dtype)
            action_ens = torch.tensor([[next_current]], dtype=input_dtype)
            
            with torch.no_grad():
                predictions_ens = active_ensemble_model(initial_state_ens, action_ens)
            
            for m in range(10):
                pred_voltage = predictions_ens[m, 0, 0].item()
                pred_temp = predictions_ens[m, 0, 1].item()
                voltage_predictions_all[m].append(pred_voltage)
                temp_predictions_all[m].append(pred_temp)
            
            median_voltage = np.median([voltage_predictions_all[m][step] for m in range(10)])
            median_temp = np.median([temp_predictions_all[m][step] for m in range(10)])
            median_voltage = np.clip(median_voltage, 2.4, 4.2)
            current_voltage = median_voltage
            current_temp = median_temp
            current_i = next_current
        
        ensemble_voltage_forecast = [np.median([voltage_predictions_all[m][step] for m in range(10)]) for step in range(steps)]
        ensemble_temp_forecast = [np.median([temp_predictions_all[m][step] for m in range(10)]) for step in range(steps)]
        
        # Calculate errors
        moe_voltage_mape = np.mean(np.abs((np.array(voltage_actual) - np.array(moe_voltage_forecast)) / np.array(voltage_actual))) * 100
        ensemble_voltage_mape = np.mean(np.abs((np.array(voltage_actual) - np.array(ensemble_voltage_forecast)) / np.array(voltage_actual))) * 100
        moe_temp_mae = np.mean(np.abs(np.array(temp_actual) - np.array(moe_temp_forecast)))
        ensemble_temp_mae = np.mean(np.abs(np.array(temp_actual) - np.array(ensemble_temp_forecast)))
        
        return jsonify({
            'status': 'success',
            'actual': {
                'voltage': voltage_actual,
                'temperature': temp_actual
            },
            'moe': {
                'voltage': moe_voltage_forecast,
                'temperature': moe_temp_forecast,
                'voltage_mape': float(moe_voltage_mape),
                'temp_mae': float(moe_temp_mae)
            },
            'ensemble': {
                'voltage': ensemble_voltage_forecast,
                'temperature': ensemble_temp_forecast,
                'voltage_mape': float(ensemble_voltage_mape),
                'temp_mae': float(ensemble_temp_mae)
            },
            'parameters': {
                'start_index': start_index,
                'requested_start_index': requested_start_index,
                'sequence_length': steps,
                'initial_voltage': initial_voltage,
                'initial_temperature': initial_temp,
                'soh': soh,
                'relative_age': relative_age,
                'current_profile': current_profile
            },
            'inference_mode': {
                'ensemble': ensemble_inference_mode,
                'ensemble_quantization_enabled': ENSEMBLE_QUANTIZATION_ENABLED
            },
            'data_split': {
                'strategy': f'block_{KIT_BLOCK_SIZE}_train_{KIT_TRAIN_BLOCK_SIZE}_val_{KIT_VAL_BLOCK_SIZE}_test_{KIT_TEST_BLOCK_SIZE}',
                'use_test_split': use_test_split,
                'requested_start_index': requested_start_index,
                'resolved_start_index': start_index,
                'test_count': count_range_positions(test_ranges),
                'test_first_index': int(test_ranges[0][0]) if test_ranges else None,
                'test_last_index': int(test_ranges[-1][1]) if test_ranges else None
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/queue_retraining_sample', methods=['POST'])
def queue_retraining_sample():
    try:
        data = request.json or {}

        start_index = data.get('start_index')
        sequence_length = data.get('sequence_length')
        thresholds = data.get('thresholds', {})
        metrics = data.get('metrics', {})
        exceeded = data.get('exceeded', {})
        models_to_train = data.get('models_to_train', [])

        if start_index is None or sequence_length is None:
            return jsonify({
                'status': 'error',
                'message': 'start_index and sequence_length are required'
            }), 400

        if not isinstance(models_to_train, list):
            models_to_train = []

        # Backward-compatible derivation when models_to_train is not explicitly sent.
        if not models_to_train and isinstance(exceeded, dict):
            moe_exceeded = exceeded.get('moe', {}) if isinstance(exceeded.get('moe', {}), dict) else {}
            ens_exceeded = exceeded.get('ensemble', {}) if isinstance(exceeded.get('ensemble', {}), dict) else {}

            if any(bool(v) for v in moe_exceeded.values()):
                models_to_train.append('moe')
            if any(bool(v) for v in ens_exceeded.values()):
                models_to_train.append('ensemble')

        # Fallback for old payload format with flat exceeded keys.
        if not models_to_train and isinstance(exceeded, dict):
            if exceeded.get('voltage_mape') or exceeded.get('temp_mae'):
                models_to_train.append('moe')

        queue_dir = os.path.join(script_dir, '..', 'retraining_queue')
        os.makedirs(queue_dir, exist_ok=True)
        queue_file = os.path.join(queue_dir, 'high_error_segments.jsonl')

        # Persist full segment context so it can be used for future training scripts.
        record = {
            'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
            'triggered_by': data.get('triggered_by', 'manual'),
            'start_index': int(start_index),
            'sequence_length': int(sequence_length),
            'thresholds': thresholds,
            'metrics': metrics,
            'exceeded': exceeded,
            'models_to_train': models_to_train,
            'parameters': data.get('parameters', {}),
            'actual': data.get('actual', {}),
            'moe': data.get('moe', {}),
            'ensemble': data.get('ensemble', {})
        }

        with open(queue_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

        # Also write per-model queue files to simplify model-specific retraining jobs.
        for model_name in models_to_train:
            if model_name not in ('moe', 'ensemble'):
                continue
            model_queue_file = os.path.join(queue_dir, f'{model_name}_high_error_segments.jsonl')
            with open(model_queue_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')

        queued_models_text = ', '.join(models_to_train).upper() if models_to_train else 'NONE'

        return jsonify({
            'status': 'success',
            'message': f"Queued segment {start_index} for retraining ({queued_models_text}).",
            'queue_file': queue_file,
            'models_to_train': models_to_train,
            'next_step': 'Use retraining_queue/high_error_segments.jsonl in your training pipeline.'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/train_queued_model', methods=['POST'])
def train_queued_model():
    try:
        data = request.json or {}
        model_name = str(data.get('model_name', '')).strip().lower()
        epochs = int(data.get('epochs', 1))
        max_samples = int(data.get('max_samples', 50))
        lr = float(data.get('learning_rate', 1e-4))
        tuning_mode = str(data.get('tuning_mode', 'adapter' if model_name == 'moe' else 'full')).strip().lower()

        rank = int(data.get('rank', moe_adapter_config.get('rank', 8)))
        alpha = float(data.get('alpha', moe_adapter_config.get('alpha', 16.0)))
        dropout = float(data.get('dropout', moe_adapter_config.get('dropout', 0.05)))
        batch_size = int(data.get('batch_size', 4))
        accumulation_steps = int(data.get('accumulation_steps', 1))

        target_modules = data.get('target_modules', moe_adapter_config.get('target_modules', DEFAULT_MOE_ADAPTER_TARGETS))
        if not isinstance(target_modules, list) or not target_modules:
            target_modules = list(DEFAULT_MOE_ADAPTER_TARGETS)

        if model_name not in ('moe', 'ensemble'):
            return jsonify({
                'status': 'error',
                'message': 'model_name must be either "moe" or "ensemble"'
            }), 400

        if tuning_mode not in ('full', 'adapter'):
            return jsonify({
                'status': 'error',
                'message': 'tuning_mode must be either "full" or "adapter"'
            }), 400

        if model_name == 'ensemble' and tuning_mode == 'adapter':
            return jsonify({
                'status': 'error',
                'message': 'Adapter tuning is currently supported only for model_name="moe"'
            }), 400

        if epochs < 1 or epochs > 20:
            return jsonify({
                'status': 'error',
                'message': 'epochs must be between 1 and 20'
            }), 400

        if max_samples < 1 or max_samples > 500:
            return jsonify({
                'status': 'error',
                'message': 'max_samples must be between 1 and 500'
            }), 400

        if lr <= 0 or lr > 0.01:
            return jsonify({
                'status': 'error',
                'message': 'learning_rate must be > 0 and <= 0.01'
            }), 400

        if rank < 1 or rank > 128:
            return jsonify({
                'status': 'error',
                'message': 'rank must be between 1 and 128'
            }), 400

        if alpha <= 0 or alpha > 1024:
            return jsonify({
                'status': 'error',
                'message': 'alpha must be > 0 and <= 1024'
            }), 400

        if dropout < 0 or dropout >= 1:
            return jsonify({
                'status': 'error',
                'message': 'dropout must be in [0, 1)'
            }), 400

        if batch_size < 1 or batch_size > 128:
            return jsonify({
                'status': 'error',
                'message': 'batch_size must be between 1 and 128'
            }), 400

        if accumulation_steps < 1 or accumulation_steps > 128:
            return jsonify({
                'status': 'error',
                'message': 'accumulation_steps must be between 1 and 128'
            }), 400

        state = get_training_state_snapshot()
        if state.get('running'):
            return jsonify({
                'status': 'busy',
                'message': f"Training already running for {state.get('model_name', 'unknown').upper()}.",
                'training_state': state
            }), 409

        worker = threading.Thread(
            target=training_worker,
            args=(
                model_name,
                epochs,
                max_samples,
                lr,
                tuning_mode,
                rank,
                alpha,
                dropout,
                batch_size,
                accumulation_steps,
                target_modules
            ),
            daemon=True
        )
        worker.start()

        return jsonify({
            'status': 'started',
            'message': f'Started background retraining for {model_name.upper()} ({tuning_mode}).',
            'requested': {
                'model_name': model_name,
                'tuning_mode': tuning_mode,
                'epochs': epochs,
                'max_samples': max_samples,
                'learning_rate': lr,
                'rank': rank,
                'alpha': alpha,
                'dropout': dropout,
                'batch_size': batch_size,
                'accumulation_steps': accumulation_steps,
                'target_modules': target_modules
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/training_status', methods=['GET'])
def training_status():
    return jsonify({
        'status': 'success',
        'training': get_training_state_snapshot()
    })


@app.route('/moe_lora_info', methods=['GET'])
def moe_lora_info():
    adapter_count = count_moe_adapter_modules(model)
    return jsonify({
        'status': 'success',
        'adapter_modules': adapter_count,
        'adapter_checkpoint_path': MOE_ADAPTER_PATH,
        'adapter_loaded': bool(adapter_count > 0),
        'adapter_merged': bool(are_moe_adapters_merged(model)),
        'message': 'Use POST /moe_lora_config with {"action":"merge"|"unmerge"} for runtime control.'
    })


@app.route('/moe_lora_config', methods=['POST'])
def moe_lora_config():
    try:
        data = request.json or {}
        action = str(data.get('action', '')).strip().lower()

        if action not in ('merge', 'unmerge'):
            return jsonify({
                'status': 'error',
                'message': 'action must be either "merge" or "unmerge"'
            }), 400

        state = get_training_state_snapshot()
        if state.get('running'):
            return jsonify({
                'status': 'busy',
                'message': 'Cannot change LoRA merge state while training is running.',
                'training_state': state
            }), 409

        with moe_adapter_lock:
            adapter_count = count_moe_adapter_modules(model)
            if adapter_count == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'No MoE adapter modules are loaded. Train/load adapter first.'
                }), 400

            if action == 'merge':
                changed = merge_moe_adapters(model)
            else:
                changed = unmerge_moe_adapters(model)

            merged_state = are_moe_adapters_merged(model)

        model.eval()
        return jsonify({
            'status': 'success',
            'action': action,
            'changed_modules': int(changed),
            'adapter_modules': int(adapter_count),
            'adapter_merged': bool(merged_state),
            'message': f'MoE LoRA adapters {"merged" if merged_state else "unmerged"}.'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/quantization_config', methods=['POST'])
def quantization_config():
    global ENSEMBLE_QUANTIZATION_ENABLED, ENSEMBLE_PRECISION_MODE

    try:
        data = request.json or {}
        requested_mode_raw = data.get('mode')
        enabled_raw = data.get('enabled')

        normalized_mode = None
        if requested_mode_raw is not None:
            normalized_mode = str(requested_mode_raw).strip().lower()
            if normalized_mode not in ('fp32', 'fp16', 'int8', 'int8_dynamic'):
                raise ValueError('mode must be one of fp32, fp16, int8, int8_dynamic')

        enabled = None
        if enabled_raw is not None:
            if isinstance(enabled_raw, bool):
                enabled = enabled_raw
            elif isinstance(enabled_raw, (int, float)):
                enabled = bool(int(enabled_raw))
            elif isinstance(enabled_raw, str):
                normalized = enabled_raw.strip().lower()
                if normalized in ('1', 'true', 'yes', 'on'):
                    enabled = True
                elif normalized in ('0', 'false', 'no', 'off'):
                    enabled = False
                else:
                    raise ValueError('enabled string must be one of true/false, 1/0, yes/no, on/off')
            else:
                raise ValueError('enabled must be bool/int/str when provided')

        with ensemble_quantization_lock:
            if normalized_mode in ('int8', 'int8_dynamic'):
                ENSEMBLE_QUANTIZATION_ENABLED = True
                ENSEMBLE_PRECISION_MODE = 'fp32'
            elif normalized_mode == 'fp16':
                ENSEMBLE_QUANTIZATION_ENABLED = False
                ENSEMBLE_PRECISION_MODE = 'fp16'
            elif normalized_mode == 'fp32':
                ENSEMBLE_QUANTIZATION_ENABLED = False
                ENSEMBLE_PRECISION_MODE = 'fp32'
            elif enabled is not None:
                ENSEMBLE_QUANTIZATION_ENABLED = enabled
                if ENSEMBLE_QUANTIZATION_ENABLED:
                    ENSEMBLE_PRECISION_MODE = 'fp32'
                elif ENSEMBLE_PRECISION_MODE not in ('fp32', 'fp16'):
                    ENSEMBLE_PRECISION_MODE = 'fp32'
            else:
                raise ValueError('provide either mode or enabled')

            refresh_ensemble_inference_model()

        return jsonify({
            'status': 'success',
            'ensemble_quantization_enabled': ENSEMBLE_QUANTIZATION_ENABLED,
            'ensemble_precision_mode': ENSEMBLE_PRECISION_MODE,
            'ensemble_inference_mode': ensemble_inference_mode,
            'available_modes': ['fp32', 'fp16', 'int8_dynamic'],
            'message': f'Ensemble runtime mode set to {ensemble_inference_mode}.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/quantization_info', methods=['GET'])
def quantization_info():
    return jsonify({
        'status': 'success',
        'ensemble_quantization_enabled': ENSEMBLE_QUANTIZATION_ENABLED,
        'ensemble_precision_mode': ENSEMBLE_PRECISION_MODE,
        'ensemble_inference_mode': ensemble_inference_mode,
        'available_modes': ['fp32', 'fp16', 'int8_dynamic'],
        'recommendation': 'Use POST /quantization_config with {"mode": "fp32"|"fp16"|"int8_dynamic"}.'
    })


@app.route('/')
def serve_index():
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/<path:path>')
def serve_frontend_assets(path):
    # Serve frontend assets and support direct URL access by falling back to index.html.
    target_path = os.path.join(frontend_dir, path)
    if os.path.exists(target_path) and os.path.isfile(target_path):
        return send_from_directory(frontend_dir, path)
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'ready'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    print("Starting Lightweight Digital Twin Forecasting Server...")
    print(f"Server running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
