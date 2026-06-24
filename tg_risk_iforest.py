import numpy as np
import os
import joblib

# IForest Engine
class TGEvaluator_IForest:
    def __init__(self, model_path="tg_risk_iforest.pkl"):
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"[TG-IForest] Loaded trained model from '{model_path}'.")
        else:
            print("[TG-IForest] No trained model found. Risk scores will default to 0.5.")

    def _extract_features(self, current_t, history):
        """Extract features"""
        if not history:
            return np.zeros(8)

        prev = history[0]  # Most recent historical login

        # Current features
        time_diff = (current_t['time'] - prev['time']) / 3600.0
        lat_diff = current_t['geo'][0] - prev['geo'][0]
        lon_diff = current_t['geo'][1] - prev['geo'][1]
        geo_dist = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
        ip_changed = 1.0 if current_t['ip'] != prev['ip'] else 0.0
        dev_changed = 1.0 if current_t['device_hash'] != prev['device_hash'] else 0.0

        # Historical features
        hist_times = []
        hist_dists = []
        hist_ip_changes = 0
        full_timeline = history[::-1]
        for i in range(1, len(full_timeline)):
            p, c = full_timeline[i - 1], full_timeline[i]
            hist_times.append((c['time'] - p['time']) / 3600.0)
            lat_d = c['geo'][0] - p['geo'][0]
            lon_d = c['geo'][1] - p['geo'][1]
            hist_dists.append(np.sqrt(lat_d ** 2 + lon_d ** 2))
            if c['ip'] != p['ip']:
                hist_ip_changes += 1

        mean_hist_time = np.mean(hist_times) if hist_times else 0.0
        mean_hist_dist = np.mean(hist_dists) if hist_dists else 0.0
        max_hist_dist = np.max(hist_dists) if hist_dists else 0.0
        hist_ip_rate = hist_ip_changes / max(len(full_timeline) - 1, 1)

        return np.array([
            time_diff, geo_dist, ip_changed, dev_changed,
            mean_hist_time, mean_hist_dist, max_hist_dist, hist_ip_rate
        ])

    def evaluate_risk(self, current_telemetry, history):
        if not history:
            return 0.05, 1.0

        if self.model is None:
            return 0.5, 0.65

        features = self._extract_features(current_telemetry, history).reshape(1, -1)

        # Run Isolation Forest
        raw_score = self.model.decision_function(features)[0]

        # Map to [0,1]
        s_anomaly = 1.0 / (1.0 + np.exp(raw_score * 5.0))
        s_anomaly = float(np.clip(s_anomaly, 0.0, 1.0))

        # Dynamic threshold
        alpha = 1.0 - (s_anomaly * 0.7)
        alpha = max(0.3, min(alpha, 1.0))

        return s_anomaly, alpha
