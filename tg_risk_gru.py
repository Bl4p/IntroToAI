import torch
import torch.nn as nn
import numpy as np
import os

# GRU Engine

class TG_RiskGRU(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=16, num_layers=1):
        super(TG_RiskGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass
        gru_out, _ = self.gru(x)
        # Get context
        context_vector = gru_out[:, -1, :]
        s_anomaly = self.regression_head(context_vector)
        return s_anomaly


class TGEvaluator_GRU:
    def __init__(self, model_path="tg_risk_gru.pt"):
        self.model = TG_RiskGRU(feature_dim=4, hidden_dim=16, num_layers=1)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"[TG-GRU] Loaded trained weights from '{model_path}'.")
        else:
            print("[TG-GRU] No trained weights found. Using random initialization.")
        self.model.eval()

    def _preprocess_telemetry(self, current_t, history):
        """Convert to tensor"""
        sequence = []
        full_timeline = history[::-1] + [current_t]

        for i in range(len(full_timeline)):
            if i == 0:
                sequence.append([0.0, 0.0, 0.0, 0.0])
                continue

            prev_log, curr_log = full_timeline[i - 1], full_timeline[i]

            time_diff = (curr_log['time'] - prev_log['time']) / 3600.0
            lat_diff = curr_log['geo'][0] - prev_log['geo'][0]
            lon_diff = curr_log['geo'][1] - prev_log['geo'][1]
            geo_dist = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
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

        # Dynamic threshold
        alpha = 1.0 - (s_anomaly * 0.7)
        alpha = max(0.3, min(alpha, 1.0))

        return s_anomaly, alpha
