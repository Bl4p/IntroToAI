import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

# Training Script for Isolation Forest Risk Engine

def generate_feature_vectors(num_normal=8000, num_anomaly=2000, max_hist_len=4):
    """
    Generates feature vectors matching the format used by TGEvaluator_IForest._extract_features().
    Features: [time_diff, geo_dist, ip_changed, dev_changed,
               mean_hist_time, mean_hist_dist, max_hist_dist, hist_ip_change_rate]
    """
    features = []
    labels = []

    # Generate NORMAL samples
    for _ in range(num_normal):
        time_diff = np.random.exponential(scale=12.0) + 0.01
        if np.random.rand() > 0.9:
            geo_dist = np.random.uniform(100.0, 5000.0)
        else:
            geo_dist = np.random.exponential(scale=5.0)
        ip_changed = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
        dev_changed = np.random.choice([0.0, 1.0], p=[0.8, 0.2])

        # Simulate normal historical aggregate stats
        mean_hist_time = np.random.uniform(2.0, 48.0)
        mean_hist_dist = np.random.exponential(scale=50.0)
        max_hist_dist = mean_hist_dist + np.random.exponential(scale=500.0)
        hist_ip_rate = np.random.uniform(0.0, 0.5)

        features.append([time_diff, geo_dist, ip_changed, dev_changed,
                         mean_hist_time, mean_hist_dist, max_hist_dist, hist_ip_rate])
        labels.append(0)

    # Generate ANOMALY samples (for evaluation only — not fed to the forest)
    for _ in range(num_anomaly):
        anomaly_type = np.random.choice(["impossible_travel", "brute_force"])
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

        mean_hist_time = np.random.uniform(2.0, 48.0)
        mean_hist_dist = np.random.exponential(scale=50.0)
        max_hist_dist = mean_hist_dist + np.random.exponential(scale=500.0)
        hist_ip_rate = np.random.uniform(0.0, 0.5)

        features.append([time_diff, geo_dist, ip_changed, dev_changed,
                         mean_hist_time, mean_hist_dist, max_hist_dist, hist_ip_rate])
        labels.append(1)

    return np.array(features), np.array(labels)


def train_iforest():
    print("[IForest Training] Generating synthetic telemetry features...")

    all_features, all_labels = generate_feature_vectors(num_normal=8000, num_anomaly=2000)

    # Isolation Forest is trained ONLY on normal data
    normal_mask = all_labels == 0
    train_features = all_features[normal_mask]

    print(f"[IForest Training] Training on {len(train_features)} normal samples...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # Expected fraction of anomalies at inference time
        max_samples='auto',
        random_state=42
    )
    model.fit(train_features)

    # Evaluate on the full dataset (normal + anomalies)
    predictions = model.predict(all_features)  # Returns 1 for normal, -1 for anomaly
    predicted_anomalies = (predictions == -1).astype(int)

    # Calculate accuracy metrics
    tp = np.sum((predicted_anomalies == 1) & (all_labels == 1))
    tn = np.sum((predicted_anomalies == 0) & (all_labels == 0))
    fp = np.sum((predicted_anomalies == 1) & (all_labels == 0))
    fn = np.sum((predicted_anomalies == 0) & (all_labels == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / len(all_labels) * 100

    print(f"\n--- Evaluation on Full Dataset ---")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

    output_filename = "tg_risk_iforest.pkl"
    joblib.dump(model, output_filename)
    print(f"\n[Success] Isolation Forest trained. Saved to '{output_filename}'")


if __name__ == "__main__":
    train_iforest()
