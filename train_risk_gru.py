import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tg_risk_gru import TG_RiskGRU

# Synthetic Telemetry Dataset
class TelemetryDataset(Dataset):
    def __init__(self, num_samples=5000, max_seq_len=5):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = []
        self.labels = []
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        for _ in range(self.num_samples):
            is_anomaly = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
            sequence = []

            sequence.append([0.0, 0.0, 0.0, 0.0])

            for _ in range(self.max_seq_len - 2):
                time_diff = np.random.uniform(1.0, 72.0)
                geo_dist = np.random.uniform(0.0, 15.0)
                ip_changed = np.random.choice([0.0, 1.0], p=[0.9, 0.1])
                dev_changed = 0.0
                sequence.append([time_diff, geo_dist, ip_changed, dev_changed])

            if is_anomaly == 1.0:
                anomaly_type = np.random.choice(["impossible_travel", "new_device_brute"])
                if anomaly_type == "impossible_travel":
                    time_diff = np.random.uniform(0.1, 0.5)
                    geo_dist = np.random.uniform(500.0, 8000.0)
                    ip_changed = 1.0
                    dev_changed = np.random.choice([0.0, 1.0], p=[0.5, 0.5])
                else:
                    time_diff = np.random.uniform(0.001, 0.01)
                    geo_dist = 0.0
                    ip_changed = 1.0
                    dev_changed = 1.0
            else:
                time_diff = np.random.uniform(1.0, 168.0)
                geo_dist = np.random.uniform(0.0, 20.0)
                ip_changed = np.random.choice([0.0, 1.0], p=[0.85, 0.15])
                dev_changed = 0.0

            sequence.append([time_diff, geo_dist, ip_changed, dev_changed])

            self.data.append(sequence)
            self.labels.append([is_anomaly])

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Training Loop

def train_gru():
    print("[GRU Training] Initializing...")

    batch_size = 64
    epochs = 15
    learning_rate = 0.001

    dataset = TelemetryDataset(num_samples=10000)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TG_RiskGRU(feature_dim=4, hidden_dim=16, num_layers=1)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Count parameters for comparison
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[GRU Training] Total parameters: {total_params}")
    print(f"[GRU Training] Beginning optimization over {epochs} epochs...\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for sequences, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * sequences.size(0)
                predicted_classes = (predictions > 0.5).float()
                correct += (predicted_classes == targets).sum().item()
                total += targets.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        accuracy = (correct / total) * 100

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

    output_filename = "tg_risk_gru.pt"
    torch.save(model.state_dict(), output_filename)
    print(f"\n[Success] GRU Risk Engine trained. Saved to '{output_filename}'")

if __name__ == "__main__":
    train_gru()
