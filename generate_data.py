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
            # Simulate an Impossible Travel or Botnet IP jump
            current_ip = "8.8.8.8" if np.random.rand() > 0.5 else "203.0.113.50"
            current_geo = (51.5074, -0.1278) # London
            current_device = "b4a1f9e2" + str(np.random.randint(1000, 9999))
            target_alpha = np.random.uniform(0.3, 0.4) # Strict zero-trust floor
        else:
            # Simulate a normal, slightly varied legitimate login
            current_ip = base_ip
            current_geo = (base_geo[0] + np.random.uniform(-0.01, 0.01), base_geo[1] + np.random.uniform(-0.01, 0.01))
            current_device = base_device
            target_alpha = np.random.uniform(0.9, 1.0) # Highly trusted

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