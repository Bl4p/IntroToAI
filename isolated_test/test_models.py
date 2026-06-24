import sys
import os
import cv2
import time

# Add parent dir to path so we can import core modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from biometric_scanner import BiometricScanner
from pfpke_crypto import PFPKECrypto
from telemetry_db import TelemetryDB

# Setup isolated environment variables (though we won't fully use app.py, good for completeness)
os.makedirs("isolated_keystore", exist_ok=True)
import app
app.KEYSTORE_DIR = "isolated_keystore"

print("--- Initializing Isolated Environment ---")
db = TelemetryDB(db_name="isolated_tg_pfpke.db")

print("\n--- Generating Synthetic Enrollment ---")
scanner = BiometricScanner()
crypto = PFPKECrypto()

# Load a real fingerprint from the workspace
fingerprint_path = os.path.join(parent_dir, "SOCOFing", "Real", "100__M_Left_index_finger.BMP")
base_fp = cv2.imread(fingerprint_path, cv2.IMREAD_GRAYSCALE)
if base_fp is None:
    print(f"Failed to load sample fingerprint: {fingerprint_path}")
    sys.exit(1)

# Perform the heavy 10-scan enrollment simulation
scans = [scanner.rotate_upright(scanner.simulate_scan(base_fp, 0.04, 3)) for _ in range(10)]
fuzzy_data = scanner.extract_fuzzy_picture(scans)
pk_f, sk = crypto.key_gen(fuzzy_data)
public_key = pk_f['pk']

enroll_telemetry = db.capture_telemetry()
db.log_attempt(public_key, enroll_telemetry, attempt_type="enrollment")

print(f"Enrolled isolated PK: {public_key[:24]}...")

print("\n--- Generating Synthetic Telemetry History ---")
# Mock 5 normal, routine logins from Los Angeles
base_time = int(time.time()) - 86400 * 5
for i in range(5):
    t = {
        "geo": (34.0522, -118.2437),
        "ip": "192.168.1.100",
        "device_hash": "a1b2c3d4e5f6",
        "time": base_time + (i * 86400)
    }
    db.log_attempt(public_key, t, attempt_type="login")

history = db.get_login_history(public_key)
print(f"Injected {len(history)} normal historical login attempts.")

# Mock a malicious, anomalous telemetry request (Impossible travel to Moscow, new device, rapid succession)
current_telemetry = {
    "geo": (55.7558, 37.6173),
    "ip": "10.0.0.99",
    "device_hash": "f6e5d4c3b2a1",
    "time": int(time.time())
}
print(f"Current Malicious Request: {current_telemetry}")

print("\n--- Evaluating Risk Engines ---")

from tg_risk_gru import TGEvaluator_GRU
from tg_risk_iforest import TGEvaluator_IForest
from tg_risk_model import TGEvaluator

gru_engine = TGEvaluator_GRU(model_path=os.path.join(parent_dir, "tg_risk_gru.pt"))
iforest_engine = TGEvaluator_IForest(model_path=os.path.join(parent_dir, "tg_risk_iforest.pkl"))
transformer_engine = TGEvaluator()

s_gru, a_gru = gru_engine.evaluate_risk(current_telemetry, history)
print(f"[GRU] Anomaly Score: {s_gru:.4f} | Alpha: {a_gru:.4f}")

s_ifo, a_ifo = iforest_engine.evaluate_risk(current_telemetry, history)
print(f"[Isolation Forest] Anomaly Score: {s_ifo:.4f} | Alpha: {a_ifo:.4f}")

s_trn, a_trn = transformer_engine.evaluate_risk(current_telemetry, history)
print(f"[Transformer] Anomaly Score: {s_trn:.4f} | Alpha: {a_trn:.4f}")

print("\n--- Test Complete ---")
