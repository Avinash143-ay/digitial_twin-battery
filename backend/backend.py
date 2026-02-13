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

# Define the Transformer model architecture
class v9_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v9_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim+1, hidden_dim)
        self.linear_in_1= nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(input_dim)  
        self.layer_norm_final = nn.LayerNorm(output_dim)  
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=2*hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))

        # Output linear layer
        self.linear_out1 = nn.Linear(2*hidden_dim, 5*output_dim)
        self.linear_out2 = nn.Linear(5*output_dim, 2*output_dim)
        self.linear_out3 = nn.Linear(2*output_dim,1*output_dim)
        self.conv1d_volt = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_temp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        actions_clone_1= actions.clone()
        actions_clone_2= actions.clone()
        actions_delta_shifted = actions_clone_2
        actions_delta_shifted[:,0:-1,:]-=actions_clone_1[:,1:,:]
        actions_delta_shifted[:,-1,:]=0

        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)
        transformer_input = torch.cat([initial_state_repeated, actions,actions_delta_shifted], dim=-1)
        transformer_input = self.linear_in(transformer_input)
        transformer_input = self.linear_in_1(transformer_input)
        
        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states_residual_1 = self.linear_out1(transformer_output)
        predicted_states_residual_1 = self.gelu(predicted_states_residual_1)
        predicted_states_residual_2 = self.linear_out2(predicted_states_residual_1)
        predicted_states_residual_2 = self.gelu(predicted_states_residual_2)
        predicted_states_residual_3 = self.linear_out3(predicted_states_residual_2)

        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states_voltage = repeated_original_voltage + predicted_states_residual_3[:,:,0]
        predicted_states_temperature = repeated_original_temprature+ (predicted_states_residual_3[:,:,1]/10)
        
        predicted_states_final = torch.stack((predicted_states_voltage, predicted_states_temperature), dim=2)

        return predicted_states_final

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# Initialize and load the Transformer model
print("Loading battery forecasting model...")
model = v9_rescaling_adaptive_TransformerModel3Decoder(
    input_dim=3+1,
    hidden_dim=150,
    output_dim=2,
    nhead=20,
    num_layers=1,
    dropout=0.1
)

# Load Transformer model weights
try:
    saved_model_path = '../models/simulator_cpu.pth'
    model_state_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    model.eval()
    print("Transformer model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load Transformer model weights: {e}")
    print("Using untrained model for demonstration purposes.")
    model.eval()

# Initialize and load the DeepEnsemble model
print("Loading DeepEnsemble model...")
ensemble_model = DeepEnsemble(num_models=10)

try:
    ensemble_path = '../models/digital_twin_simpler.pt'
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'ready'})

if __name__ == '__main__':
    print("Starting Lightweight Digital Twin Forecasting Server...")
    print("Server running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
