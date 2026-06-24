import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Add parent dir to path so we can import core modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from tg_risk_gru import TGEvaluator_GRU
from tg_risk_iforest import TGEvaluator_IForest
from tg_risk_model import TGEvaluator

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the threshold where FAR (FPR) is closest to FRR (FNR)
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def generate_test_dataset(num_samples=1000):
    dataset = []
    labels = [] # 1.0 for anomaly, 0.0 for normal
    
    base_time = int(time.time()) - 30 * 86400
    base_lat, base_lon = 34.0522, -118.2437 # LA
    base_ip = "192.168.1.100"
    base_dev = "a1b2c3d4e5f6"
    
    print(f"Generating {num_samples} synthetic telemetry test samples...")
    
    for i in range(num_samples):
        # 20% anomalies
        is_anomaly = 1.0 if np.random.rand() < 0.2 else 0.0
        
        # Build normal history of length 3 to 7
        history_len = np.random.randint(3, 8)
        history = []
        for j in range(history_len):
            history.append({
                "geo": (base_lat + np.random.normal(0, 0.05), base_lon + np.random.normal(0, 0.05)),
                "ip": base_ip if np.random.rand() < 0.9 else "192.168.1.101",
                "device_hash": base_dev,
                "time": base_time + (j * 86400) + np.random.randint(-3600, 3600)
            })
            
        # Generate the current request
        current_time = history[-1]["time"] + np.random.randint(3600, 86400)
        
        if is_anomaly:
            anomaly_type = np.random.choice(["impossible_travel", "new_device_rapid"])
            if anomaly_type == "impossible_travel":
                # Very fast, very far
                current_time = history[-1]["time"] + np.random.randint(60, 600) # 1-10 mins
                current_telemetry = {
                    "geo": (base_lat + np.random.uniform(10, 50), base_lon + np.random.uniform(10, 50)),
                    "ip": "8.8.8.8",
                    "device_hash": base_dev if np.random.rand() < 0.5 else "malicious_dev_99",
                    "time": current_time
                }
            else:
                # Same place, brand new device rapidly requesting
                current_time = history[-1]["time"] + np.random.randint(1, 60) # 1-60 seconds
                current_telemetry = {
                    "geo": history[-1]["geo"],
                    "ip": "8.8.8.8",
                    "device_hash": "hacker_device_x",
                    "time": current_time
                }
        else:
            current_telemetry = {
                "geo": (base_lat + np.random.normal(0, 0.05), base_lon + np.random.normal(0, 0.05)),
                "ip": base_ip if np.random.rand() < 0.9 else "192.168.1.101",
                "device_hash": base_dev,
                "time": current_time
            }
            
        dataset.append((current_telemetry, history))
        labels.append(is_anomaly)
        
    return dataset, labels

def evaluate_engine(engine_name, engine, dataset, labels):
    print(f"Evaluating {engine_name}...")
    scores = []
    alphas = []
    latencies = []
    
    for (current_telemetry, history) in dataset:
        start_time = time.time()
        s_anomaly, alpha = engine.evaluate_risk(current_telemetry, history)
        latency = (time.time() - start_time) * 1000 # convert to ms
        latencies.append(latency)
        scores.append(s_anomaly)
        alphas.append(alpha)
        
    y_true = np.array(labels)
    y_scores = np.array(scores)
    y_alphas = np.array(alphas)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    eer = calculate_eer(y_true, y_scores)
    
    # Calculate Alpha spread
    alpha_normal = np.mean(y_alphas[y_true == 0.0])
    alpha_anomaly = np.mean(y_alphas[y_true == 1.0])
    alpha_spread = alpha_normal - alpha_anomaly
    
    mean_latency = np.mean(latencies)

    return {
        "Engine": engine_name,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "EER": eer,
        "Mean Latency (ms)": mean_latency,
        "Alpha Spread": alpha_spread,
        "Mean Alpha (Normal)": alpha_normal,
        "Mean Alpha (Anomaly)": alpha_anomaly
    }

def main():
    print("--- Risk Engine Evaluation Script ---")
    dataset, labels = generate_test_dataset(num_samples=1000)
    
    # Initialize engines
    gru_engine = TGEvaluator_GRU(model_path=os.path.join(parent_dir, "tg_risk_gru.pt"))
    iforest_engine = TGEvaluator_IForest(model_path=os.path.join(parent_dir, "tg_risk_iforest.pkl"))
    transformer_engine = TGEvaluator() # Transformer is initialized from scratch since it doesn't load automatically
    
    results = []
    results.append(evaluate_engine("TG-GRU", gru_engine, dataset, labels))
    results.append(evaluate_engine("Isolation Forest", iforest_engine, dataset, labels))
    results.append(evaluate_engine("Transformer", transformer_engine, dataset, labels))
    
    # Output results
    df = pd.DataFrame(results)
    
    csv_path = "model_evaluation_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n--- Evaluation Results ---")
    print(df.to_string(index=False))
    print(f"\n[Success] Metrics exported to '{os.path.abspath(csv_path)}'")

if __name__ == "__main__":
    main()
