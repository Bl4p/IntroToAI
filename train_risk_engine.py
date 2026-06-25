import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tg_risk_model import TG_RiskTransformer


# Data Generator
class TelemetryDataset(Dataset):
    def __init__(self, num_samples=5000, max_seq_len=5):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = []
        self.labels = []
        
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Build dataset"""
        for _ in range(self.num_samples):
            # Select behavior
            is_anomaly = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
            sequence = []
            
            # Baseline entry is always clean (delta is zero)
            sequence.append([0.0, 0.0, 0.0, 0.0])
            
            # Populate historical logs (indices 1 to max_seq_len - 2)
            for _ in range(self.max_seq_len - 2):
                time_diff = np.random.exponential(scale=12.0) + 0.01
                if np.random.rand() > 0.9:
                    geo_dist = np.random.uniform(100.0, 5000.0)
                else:
                    geo_dist = np.random.exponential(scale=5.0)
                ip_changed = np.random.choice([0.0, 1.0], p=[0.8, 0.2])
                dev_changed = np.random.choice([0.0, 1.0], p=[0.9, 0.1])
                sequence.append([time_diff, geo_dist, ip_changed, dev_changed])
            
            # Populate the final entry
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

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Train Loop
def train_engine():
    print("[Training] Initializing Telemetry Pipeline...")
    
    # Hyperparameters
    batch_size = 64
    epochs = 15
    learning_rate = 0.001
    
    # Init dataset
    dataset = TelemetryDataset(num_samples=10000)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Init model
    model = TG_RiskTransformer(feature_dim=4, d_model=32, nhead=4, num_layers=2)
    
    # Configure optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"[Training] Beginning optimization loop over {epochs} epochs...\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * sequences.size(0)
                
                # Accuracy metrics
                predicted_classes = (predictions > 0.5).float()
                correct += (predicted_classes == targets).sum().item()
                total += targets.size(0)
                
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        accuracy = (correct / total) * 100
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

    # Save weights
    output_filename = "tg_risk_model.pt"
    torch.save(model.state_dict(), output_filename)
    print(f"\n[Success] Risk Engine fully trained. Saved model parameters to '{output_filename}'")

if __name__ == "__main__":
    train_engine()