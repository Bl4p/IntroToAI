import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader, Dataset
from tg_risk_gru import TG_RiskGRU
from tg_risk_model import TG_RiskTransformer

class TestTelemetryDataset(Dataset):
    def __init__(self, num_samples=5000, max_seq_len=5):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = []
        self.labels = []
        self.iforest_features = []
        
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        for _ in range(self.num_samples):
            is_anomaly = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
            sequence = []
            sequence.append([0.0, 0.0, 0.0, 0.0])

            hist_times = []
            hist_dists = []
            hist_ip_changes = 0

            for _ in range(self.max_seq_len - 2):
                time_diff = np.random.exponential(scale=12.0) + 0.01
                if np.random.rand() > 0.9:
                    geo_dist = np.random.uniform(100.0, 5000.0)
                else:
                    geo_dist = np.random.exponential(scale=5.0)
                ip_changed = np.random.choice([0.0, 1.0], p=[0.8, 0.2])
                dev_changed = np.random.choice([0.0, 1.0], p=[0.9, 0.1])
                
                sequence.append([time_diff, geo_dist, ip_changed, dev_changed])
                hist_times.append(time_diff)
                hist_dists.append(geo_dist)
                hist_ip_changes += ip_changed

            if is_anomaly == 1.0:
                anomaly_type = np.random.choice(["impossible_travel", "new_device_brute"])
                if anomaly_type == "impossible_travel":
                    time_diff = np.random.uniform(0.1, 10.0)
                    geo_dist = np.random.uniform(100.0, 5000.0)
                    ip_changed = np.random.choice([0.0, 1.0], p=[0.1, 0.9])
                    dev_changed = np.random.choice([0.0, 1.0], p=[0.5, 0.5])
                else:
                    time_diff = np.random.uniform(0.01, 1.0)
                    geo_dist = np.random.exponential(scale=5.0)
                    ip_changed = np.random.choice([0.0, 1.0], p=[0.5, 0.5])
                    dev_changed = np.random.choice([0.0, 1.0], p=[0.2, 0.8])
            else:
                time_diff = np.random.exponential(scale=12.0) + 0.01
                if np.random.rand() > 0.9:
                    geo_dist = np.random.uniform(100.0, 5000.0)
                else:
                    geo_dist = np.random.exponential(scale=5.0)
                ip_changed = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
                dev_changed = np.random.choice([0.0, 1.0], p=[0.8, 0.2])

            sequence.append([time_diff, geo_dist, ip_changed, dev_changed])
            
            self.data.append(sequence)
            self.labels.append([is_anomaly])
            
            # Compute aggregate features for iforest
            mean_hist_time = np.mean(hist_times) if hist_times else 0.0
            mean_hist_dist = np.mean(hist_dists) if hist_dists else 0.0
            max_hist_dist = np.max(hist_dists) if hist_dists else 0.0
            hist_ip_rate = hist_ip_changes / len(hist_times) if hist_times else 0.0
            
            self.iforest_features.append([
                time_diff, geo_dist, ip_changed, dev_changed,
                mean_hist_time, mean_hist_dist, max_hist_dist, hist_ip_rate
            ])

        self.data_tensor = torch.tensor(self.data, dtype=torch.float32)
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.float32)
        self.iforest_features = np.array(self.iforest_features)
        self.iforest_labels = np.array(self.labels).flatten()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.labels_tensor[idx]

def evaluate_models():
    print("Generating 5000 entirely new synthetic test samples...")
    test_dataset = TestTelemetryDataset(num_samples=5000)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 1. GRU
    print("\n--- Testing GRU Risk Engine ---")
    gru_model = TG_RiskGRU(feature_dim=4, hidden_dim=16, num_layers=1)
    gru_model.load_state_dict(torch.load("tg_risk_gru.pt"))
    gru_model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for seqs, targets in test_loader:
            preds = gru_model(seqs)
            pred_classes = (preds > 0.5).float()
            correct += (pred_classes == targets).sum().item()
            total += targets.size(0)
    print(f"GRU Accuracy: {(correct/total)*100:.2f}%")

    # 2. Transformer
    print("\n--- Testing Transformer Risk Engine ---")
    tf_model = TG_RiskTransformer(feature_dim=4, d_model=32, nhead=4, num_layers=2)
    tf_model.load_state_dict(torch.load("tg_risk_model.pt"))
    tf_model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for seqs, targets in test_loader:
            preds = tf_model(seqs)
            pred_classes = (preds > 0.5).float()
            correct += (pred_classes == targets).sum().item()
            total += targets.size(0)
    print(f"Transformer Accuracy: {(correct/total)*100:.2f}%")
    
    # 3. Isolation Forest
    print("\n--- Testing Isolation Forest ---")
    iforest_model = joblib.load("tg_risk_iforest.pkl")
    preds = iforest_model.predict(test_dataset.iforest_features)
    # iforest returns -1 for anomaly, 1 for normal
    pred_anomalies = (preds == -1).astype(int)
    
    tp = np.sum((pred_anomalies == 1) & (test_dataset.iforest_labels == 1))
    tn = np.sum((pred_anomalies == 0) & (test_dataset.iforest_labels == 0))
    fp = np.sum((pred_anomalies == 1) & (test_dataset.iforest_labels == 0))
    fn = np.sum((pred_anomalies == 0) & (test_dataset.iforest_labels == 1))
    
    accuracy = (tp + tn) / len(test_dataset.iforest_labels) * 100
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    
    print(f"IForest Accuracy: {accuracy:.2f}%")
    print(f"IForest Precision: {precision:.4f}")
    print(f"IForest Recall: {recall:.4f}")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

if __name__ == '__main__':
    evaluate_models()
