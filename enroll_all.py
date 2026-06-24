import numpy as np
import cv2
import os
import time
import glob
from biometric_scanner import BiometricScanner
from telemetry_db import TelemetryDB
from pfpke_crypto import PFPKECrypto

def enroll_all():
    print("--- Bulk Enrollment Initialization ---")
    scanner = BiometricScanner()
    db = TelemetryDB()
    crypto = PFPKECrypto()

    real_fingerprints_dir = "SOCOFing/Real"
    
    if not os.path.exists(real_fingerprints_dir):
        print(f"Error: Directory '{real_fingerprints_dir}' not found.")
        return

    # Find all BMP files in the Real fingerprints directory and limit to 20
    fingerprint_files = glob.glob(os.path.join(real_fingerprints_dir, "*.BMP"))[:20]
    total_files = len(fingerprint_files)
    
    print(f"Found {total_files} fingerprints to enroll.")

    for i, file_path in enumerate(fingerprint_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{total_files}] Enrolling {filename}...")
        
        # 1. Load Image
        base_fp = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if base_fp is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        # 2. Enrollment Phase
        scans = [scanner.rotate_upright(scanner.simulate_scan(base_fp, 0.04, 3)) for _ in range(10)]
        fuzzy_data = scanner.extract_fuzzy_picture(scans)
        
        # Generate keys
        pk_f, secret_key = crypto.key_gen(fuzzy_data)
        public_key_string = pk_f['pk']
        
        print(f"Public Key Generated: {public_key_string[:16]}...")

        # Log enrollment
        enrollment_telemetry = db.capture_telemetry()
        db.log_attempt(public_key_string, enrollment_telemetry, attempt_type="enrollment")

    print("\n--- Bulk Enrollment Complete ---")

if __name__ == "__main__":
    enroll_all()
