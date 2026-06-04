import numpy as np
import cv2
import os
import time
from biometric_scanner import BiometricScanner
from telemetry_db import TelemetryDB
from pfpke_crypto import PFPKECrypto
from tg_risk_model import TGEvaluator

def main():
    print("--- TG-PFPKE System Initialization ---")
    scanner = BiometricScanner()
    db = TelemetryDB()
    crypto = PFPKECrypto()
    llm = TGEvaluator()

    # 1. Load Image
    socofing_image_path = "SOCOFing/Real/1__M_Left_index_finger.BMP" 
    if not os.path.exists(socofing_image_path):
        print(f"Error: Could not find '{socofing_image_path}'.")
        return
    base_fp = cv2.imread(socofing_image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Enrollment Phase
    print("\n[Enrollment] Scanning biometrics...")
    scans = [scanner.rotate_upright(scanner.simulate_scan(base_fp, 0.04, 3)) for _ in range(10)]
    fuzzy_data = scanner.extract_fuzzy_picture(scans)
    
    pk_f, secret_key = crypto.key_gen(fuzzy_data)
    public_key_string = pk_f['pk']
    print(f"Public Key Generated: {public_key_string[:16]}...")

    enrollment_telemetry = db.capture_telemetry()
    db.log_attempt(public_key_string, enrollment_telemetry, attempt_type="enrollment")

    # 3. Encryption Phase
    print("\n--- Encryption Phase ---")
    secret_message = "Corporate_Financial_Report_Q3"
    ciphertext = crypto.encrypt(pk_f, secret_message)
    print("Ciphertext Package Generated (svk, CT, sigma).")

    # 4. Decryption Phase (Legitimate Authentication Attempt)
    print("\n--- Decryption Phase (Legitimate Login) ---")
    login_scan_raw = scanner.simulate_scan(base_fp, noise_variance=0.06, blur_kernel_size=3)
    login_scan_aligned = scanner.rotate_upright(login_scan_raw)
    login_fuzzy_data = scanner.extract_fuzzy_picture([login_scan_aligned])

    current_telemetry = db.capture_telemetry()
    history = db.get_login_history(public_key_string)
    
    # 5. Pre-Authentication Risk Evaluation (Will hit 0.05 default here)
    s_anomaly, alpha = llm.evaluate_risk(current_telemetry, history)
    print(f"[TG-LLM] Environmental Anomaly Score: {s_anomaly:.4f}")
    
    if s_anomaly > 0.95:
        print("\n[Denied] CRITICAL THREAT DETECTED.")
        return 

    print(f"[TG-LLM] Active Biometric Threshold Modifier (alpha): {alpha:.4f}")
    db.log_attempt(public_key_string, current_telemetry, attempt_type="login")

    # 6. Cryptographic Reconstruction
    try:
        decrypted_message = crypto.decrypt(pk_f, login_fuzzy_data, ciphertext, alpha=alpha)
        print(f"[Success] Decryption Successful! Recovered Message: {decrypted_message}")
    except ValueError as e:
        print(f"[Denied] Authentication Denied: {e}")

    # ==========================================
    # 7. IMPOSSIBLE TRAVEL ATTACK SIMULATION
    # ==========================================
    print("\n--- Decryption Phase (Simulated Attack) ---")
    time.sleep(1) # Wait 1 second to ensure distinct timestamp

    # Spoof telemetry to look like a malicious actor in London right after you logged in from Hanoi
    attack_telemetry = db.capture_telemetry()
    attack_telemetry['ip'] = "8.8.8.8"           
    attack_telemetry['geo'] = (51.5074, -0.1278) 
    
    # Retrieve the history (This now contains your legitimate login from above!)
    history = db.get_login_history(public_key_string)
    
    # Feed the attack sequence into the TRAINED PyTorch Transformer
    s_anomaly, alpha = llm.evaluate_risk(attack_telemetry, history)
    print(f"[TG-LLM] Environmental Anomaly Score: {s_anomaly:.4f}")
    
    # The Hard Gate (Kill Switch) should trigger here
    if s_anomaly > 0.95:
        print("\n[Denied] CRITICAL THREAT DETECTED. Impossible travel sequence.")
        print("Dropping connection. Cryptography engine bypassed.")
        return 

    print(f"[TG-LLM] Active Biometric Threshold Modifier (alpha): {alpha:.4f}")
    
    try:
        decrypted_message = crypto.decrypt(pk_f, login_fuzzy_data, ciphertext, alpha=alpha)
        print(f"[Success] Decryption Successful! Recovered Message: {decrypted_message}")
    except ValueError as e:
        print(f"[Denied] Authentication Denied: {e}")

if __name__ == "__main__":
    main()

    #