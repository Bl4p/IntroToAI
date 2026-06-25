import pandas as pd
import numpy as np
import math

def generate_synthetic_prompts(num_samples=5000):
    print(f"Generating {num_samples} synthetic training sequences for TG-LLM...")
    data = []

    for _ in range(num_samples):
        # 30% chance to generate a malicious attack, 70% chance for normal behavior
        is_attack = np.random.choice([True, False], p=[0.3, 0.7])
        
        # Baseline IP and Geo for the user's history
        base_ip = f"192.168.1.{np.random.randint(2, 20)}"
        base_geo = (21.0285 + np.random.uniform(-0.05, 0.05), 105.8542 + np.random.uniform(-0.05, 0.05))
        base_device = "6f5129ac" + str(np.random.randint(1000, 9999))
        
        history_text = "[HISTORY]\n"
        for i in range(3): # Simulate 3 past normal logs
            history_text += f"Log {i+1}: IP address {base_ip}, Location coordinates {base_geo[0]:.4f},{base_geo[1]:.4f}, Hardware Device {base_device[:8]}.\n"

        if is_attack:
            # Simulate an Impossible Travel or Botnet IP jump with realistic variations
            current_ip = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
            # Sometimes attack is close, sometimes far
            current_geo = (base_geo[0] + np.random.uniform(-30, 30), base_geo[1] + np.random.uniform(-30, 30))
            # Attackers might hijack a session/device occasionally
            current_device = base_device if np.random.rand() > 0.85 else "b4a1f9e2" + str(np.random.randint(1000, 9999))
            target_alpha = np.random.uniform(0.1, 0.6) # Target alpha spans a wider range
        else:
            # Simulate a normal login, but with occasional real-world noise (e.g., VPNs, new IPs)
            current_ip = base_ip if np.random.rand() > 0.15 else f"192.168.1.{np.random.randint(2, 255)}"
            current_geo = (base_geo[0] + np.random.uniform(-0.5, 0.5), base_geo[1] + np.random.uniform(-0.5, 0.5))
            # VPN users might jump further randomly, simulating false positives in the training set
            if np.random.rand() > 0.95:
                current_geo = (base_geo[0] + np.random.uniform(-10, 10), base_geo[1] + np.random.uniform(-10, 10))
            # User might log in from a new device occasionally
            current_device = base_device if np.random.rand() > 0.05 else "6f5129ac" + str(np.random.randint(1000, 9999))
            target_alpha = np.random.uniform(0.65, 1.0)

        # Calculate haversine distance for the prompt
        lat1, lon1, lat2, lon2 = map(math.radians, [base_geo[0], base_geo[1], current_geo[0], current_geo[1]])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        distance = 2 * 6371 * math.asin(math.sqrt(a))

        prompt = "Analyze the following user login sequence for security threats:\n"
        prompt += history_text
        prompt += f"[ANALYSIS] The new login is {distance:.2f} kilometers away from the previous location.\n\n"
        prompt += f"[CURRENT ATTEMPT] IP address {current_ip}, Location coordinates {current_geo[0]:.4f},{current_geo[1]:.4f}, Hardware Device {current_device[:8]}."

        data.append({"text": prompt, "label": round(target_alpha, 4)})

    df = pd.DataFrame(data)
    df.to_csv("telemetry_data.csv", index=False)
    print("Successfully saved 'telemetry_data.csv'. Ready for training!")

if __name__ == "__main__":
    generate_synthetic_prompts()