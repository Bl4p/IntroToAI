import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tg_risk_model import TG_RiskTransformer

# =====================================================================
# 1. Synthetic Telemetry Dataset Generator
# =====================================================================
class TelemetryDataset(Dataset):
    def __init__(self, num_samples=5000, max_seq_len=5):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.data = []
        self.labels = []
        
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """
        Generates realistic chronological sequences of network behaviors.
        Features per log: [time_diff_hours, geo_dist_km, ip_changed, dev_changed]
        """
        for _ in range(self.num_samples):
            # Choose randomly if this sequence ends in an anomaly (1.0) or normal behavior (0.0)
            is_anomaly = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
            sequence = []
            
            # Baseline entry is always clean (delta is zero)
            sequence.append([0.0, 0.0, 0.0, 0.0])
            
            # Populate historical logs (indices 1 to max_seq_len - 2)
            for _ in range(self.max_seq_len - 2):
                # Normal minor fluctuations in standard user history
                time_diff = np.random.uniform(1.0, 72.0)
                geo_dist = np.random.uniform(0.0, 15.0) 
                ip_changed = np.random.choice([0.0, 1.0], p=[0.9, 0.1])
                dev_changed = 0.0 # Device rarely changes in normal historical sequences
                sequence.append([time_diff, geo_dist, ip_changed, dev_changed])
            
            # Populate the final entry (The current login attempt under evaluation)
            if is_anomaly == 1.0:
                # Malicious behaviors (e.g., credential stuffing, impossible travel)
                anomaly_type = np.random.choice(["impossible_travel", "new_device_brute"])
                if anomaly_type == "impossible_travel":
                    time_diff = np.random.uniform(0.1, 0.5)  # 5 to 30 minutes
                    geo_dist = np.random.uniform(500.0, 8000.0) # Thousands of kilometers away
                    ip_changed = 1.0
                    dev_changed = np.random.choice([0.0, 1.0], p=[0.5, 0.5])
                else: 
                    # New device, rapid successive requests (credential brute-force)
                    time_diff = np.random.uniform(0.001, 0.01) # Fractions of an hour
                    geo_dist = 0.0
                    ip_changed = 1.0
                    dev_changed = 1.0
            else:
                # Normal benign current login attempt
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

# =====================================================================
# 2. Execution Training Loop
# =====================================================================
def train_engine():
    print("[Training] Initializing Telemetry Pipeline...")
    
    # Hyperparameters
    batch_size = 64
    epochs = 15
    learning_rate = 0.001
    
    # Initialize components (generating 10,000 sequences)
    dataset = TelemetryDataset(num_samples=10000)
    
    # Split into train/validation (80% Train / 20% Validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the untrained architecture from your tg_risk_model package
    model = TG_RiskTransformer(feature_dim=4, d_model=32, nhead=4, num_layers=2)
    
    # Binary Cross-Entropy is ideal for 0.0 to 1.0 probability outputs
    criterion = nn.BCELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    print(f"[Training] Beginning optimization loop over {epochs} epochs...\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass through self-attention layers
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            # Backpropagation and weight updates
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
            
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * sequences.size(0)
                
                # Turn continuous probability into binary classification for accuracy metrics
                predicted_classes = (predictions > 0.5).float()
                correct += (predicted_classes == targets).sum().item()
                total += targets.size(0)
                
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        accuracy = (correct / total) * 100
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

    # Serialize the optimized matrix weights
    output_filename = "tg_risk_transformer.pt"
    torch.save(model.state_dict(), output_filename)
    print(f"\n[Success] Risk Engine fully trained. Saved model parameters to '{output_filename}'")

if __name__ == "__main__":
    train_engine()