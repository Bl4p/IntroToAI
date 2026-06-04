import torch
import torch.nn as nn
import numpy as np

class TG_RiskTransformer(nn.Module):
    def __init__(self, feature_dim=4, d_model=32, nhead=4, num_layers=2):
        super(TG_RiskTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=128, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Outputs S_anomaly strictly bounded between 0.0 and 1.0
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_out = self.transformer(embedded)
        context_vector = transformer_out[:, -1, :] 
        s_anomaly = self.regression_head(context_vector)
        return s_anomaly

class TGEvaluator:
    def __init__(self):
        self.model = TG_RiskTransformer()
        self.model.eval() 

    def _preprocess_telemetry(self, current_t, history):
        sequence = []
        full_timeline = history[::-1] + [current_t] 
        
        for i in range(len(full_timeline)):
            if i == 0:
                sequence.append([0.0, 0.0, 0.0, 0.0])
                continue
                
            prev_log, curr_log = full_timeline[i-1], full_timeline[i]
            
            time_diff = (curr_log['time'] - prev_log['time']) / 3600.0
            lat_diff = curr_log['geo'][0] - prev_log['geo'][0]
            lon_diff = curr_log['geo'][1] - prev_log['geo'][1]
            geo_dist = np.sqrt(lat_diff**2 + lon_diff**2)
            ip_changed = 1.0 if curr_log['ip'] != prev_log['ip'] else 0.0
            dev_changed = 1.0 if curr_log['device_hash'] != prev_log['device_hash'] else 0.0
            
            sequence.append([time_diff, geo_dist, ip_changed, dev_changed])
            
        return torch.tensor([sequence], dtype=torch.float32)

    def evaluate_risk(self, current_telemetry, history):
        if not history:
            return 0.05, 1.0
            
        x_tensor = self._preprocess_telemetry(current_telemetry, history)
        
        with torch.no_grad():
            s_anomaly = self.model(x_tensor).item()
            
        # Calculate alpha based on the risk score, bounded at 0.3
        alpha = 1.0 - (s_anomaly * 0.7)
        alpha = max(0.3, min(alpha, 1.0))
        
        return s_anomaly, alpha