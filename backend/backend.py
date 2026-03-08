from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

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
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
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

# Initialize and load the DeepEnsemble model
print("Loading DeepEnsemble model...")
ensemble_model = DeepEnsemble(num_models=10)

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

device = torch.device("cpu")

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
            
            # Prepare initial state: [relative_age, voltage, temp, current_at_t0]
            initial_state = torch.tensor([[current_age, current_voltage, current_temp, current_i]], 
                                        dtype=torch.float32)
            # Action is the next current (no negation - was wrong)
            action = torch.tensor([[next_current]], dtype=torch.float32)
            
            # Get ensemble predictions
            with torch.no_grad():
                predictions = ensemble_model(initial_state, action)  # [num_models, batch, 2]
            
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

@app.route('/compare_with_dataset', methods=['POST'])
def compare_with_dataset():
    try:
        import pandas as pd
        data = request.json
        
        # Get parameters
        start_index = int(data.get('start_index', 10000))
        sequence_length = int(data.get('sequence_length', 75))
        
        # Load segment from KIT dataset
        dataset_path = os.path.join(script_dir, '..', 'data', 'cell_log_age_2s_P065_1_S01_C03', 'cell_log_age_2s_P065_1_S01_C03.csv')
        
        if not os.path.exists(dataset_path):
            return jsonify({
                'status': 'error',
                'message': f'Dataset not found at {dataset_path}'
            }), 404
        
        # Read the specific segment
        df = pd.read_csv(dataset_path, sep=';', skiprows=range(1, start_index), nrows=sequence_length+1)
        
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
        voltage_predictions_all = [[] for _ in range(10)]
        temp_predictions_all = [[] for _ in range(10)]
        
        current_voltage = initial_voltage
        current_temp = initial_temp
        current_age = relative_age
        current_i = 0.0
        
        for step in range(steps):
            next_current = current_profile[step]
            initial_state_ens = torch.tensor([[current_age, current_voltage, current_temp, current_i]], dtype=torch.float32)
            action_ens = torch.tensor([[next_current]], dtype=torch.float32)
            
            with torch.no_grad():
                predictions_ens = ensemble_model(initial_state_ens, action_ens)
            
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
                'sequence_length': steps,
                'initial_voltage': initial_voltage,
                'initial_temperature': initial_temp,
                'soh': soh
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'ready'})

if __name__ == '__main__':
    print("Starting Lightweight Digital Twin Forecasting Server...")
    print("Server running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
