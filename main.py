import argparse
import numpy as np
import cv2
import os
import pickle
import glob
from biometric_scanner import BiometricScanner
from telemetry_db import TelemetryDB
from pfpke_crypto import PFPKECrypto

KEYSTORE_DIR = "keystore"

def _load_engine(engine_name):
    """Load the selected risk evaluation engine."""
    if engine_name == "gru":
        from tg_risk_gru import TGEvaluator_GRU
        return TGEvaluator_GRU()
    elif engine_name == "iforest":
        from tg_risk_iforest import TGEvaluator_IForest
        return TGEvaluator_IForest()
    else:
        from tg_risk_model import TGEvaluator
        return TGEvaluator()

def _keystore_path(user_id):
    """Return the file path for a given user's keystore."""
    return os.path.join(KEYSTORE_DIR, f"{user_id}.pkl")

def _save_keystore(user_id, data):
    os.makedirs(KEYSTORE_DIR, exist_ok=True)
    path = _keystore_path(user_id)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path

def _load_keystore(user_id):
    path = _keystore_path(user_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def _list_users():
    """List all enrolled user IDs."""
    if not os.path.exists(KEYSTORE_DIR):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(KEYSTORE_DIR) if f.endswith(".pkl")]

def _load_fingerprint(image_path):
    """Load and validate a fingerprint image."""
    if not os.path.exists(image_path):
        print(f"Error: Could not find '{image_path}'.")
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return None
    return img

# Enrollment

def enroll(args):
    print("--- Enrollment Phase ---")
    scanner = BiometricScanner()
    db = TelemetryDB()
    crypto = PFPKECrypto()

    base_fp = _load_fingerprint(args.image)
    if base_fp is None:
        return

    # Simulate multiple scans for a robust fuzzy picture
    scans = [scanner.rotate_upright(scanner.simulate_scan(base_fp, 0.04, 3)) for _ in range(10)]
    fuzzy_data = scanner.extract_fuzzy_picture(scans)

    pk_f, secret_key = crypto.key_gen(fuzzy_data)
    public_key_string = pk_f['pk']

    # Use the fingerprint filename as the user ID for easy lookup
    user_id = os.path.splitext(os.path.basename(args.image))[0]

    keystore_data = {
        "user_id": user_id,
        "pk_f": pk_f,
        "sk": secret_key,
        "public_key": public_key_string,
        "image_path": args.image,
    }
    path = _save_keystore(user_id, keystore_data)

    # Log enrollment telemetry
    telemetry = db.capture_telemetry()
    db.log_attempt(public_key_string, telemetry, attempt_type="enrollment")

    print(f"User ID:     {user_id}")
    print(f"Public Key:  {public_key_string[:16]}...")
    print(f"Keystore:    {path}")
    print("[Success] Fingerprint enrolled.")

# Encryption

def encrypt(args):
    print("--- Encryption Phase ---")
    crypto = PFPKECrypto()

    # Load the enrolled keystore by user ID
    keystore = _load_keystore(args.user_id)
    if keystore is None:
        print(f"Error: No enrollment found for user '{args.user_id}'.")
        print(f"Available users: {_list_users() or 'none — run enroll first'}")
        return

    pk_f = keystore['pk_f']

    # Encrypt the message
    ciphertext = crypto.encrypt(pk_f, args.message)

    # Save ciphertext alongside the keystore
    keystore['ciphertext'] = ciphertext
    _save_keystore(args.user_id, keystore)

    print(f"User ID:     {args.user_id}")
    print(f"Message:     \"{args.message}\"")
    print("[Success] Ciphertext package generated (svk, CT, sigma).")

# Decryption

def decrypt(args):
    print("--- Decryption Phase ---")
    scanner = BiometricScanner()
    db = TelemetryDB()
    crypto = PFPKECrypto()
    evaluator = _load_engine(args.engine)

    # Load the enrolled keystore by user ID
    keystore = _load_keystore(args.user_id)
    if keystore is None:
        print(f"Error: No enrollment found for user '{args.user_id}'.")
        print(f"Available users: {_list_users() or 'none — run enroll first'}")
        return
    if 'ciphertext' not in keystore:
        print(f"Error: No encrypted message found for user '{args.user_id}'. Run 'encrypt' first.")
        return

    pk_f = keystore['pk_f']
    ciphertext = keystore['ciphertext']
    public_key_string = keystore['public_key']
    image_path = keystore['image_path']

    # Load the enrolled fingerprint for a fresh authentication scan
    base_fp = _load_fingerprint(image_path)
    if base_fp is None:
        return

    # Simulate a fresh login scan (slightly noisy, as in real life)
    login_scan = scanner.rotate_upright(scanner.simulate_scan(base_fp, noise_variance=0.06, blur_kernel_size=3))
    login_fuzzy_data = scanner.extract_fuzzy_picture([login_scan])

    # Risk evaluation
    current_telemetry = db.capture_telemetry()
    history = db.get_login_history(public_key_string)

    s_anomaly, alpha = evaluator.evaluate_risk(current_telemetry, history)
    print(f"[Risk Engine ({args.engine})] Anomaly Score: {s_anomaly:.4f}")

    if s_anomaly > 0.95:
        print("[Denied] CRITICAL THREAT DETECTED. Connection dropped.")
        return

    print(f"[Risk Engine ({args.engine})] Biometric Threshold Modifier (alpha): {alpha:.4f}")
    db.log_attempt(public_key_string, current_telemetry, attempt_type="login")

    # Cryptographic reconstruction
    try:
        decrypted_message = crypto.decrypt(pk_f, login_fuzzy_data, ciphertext, alpha=alpha)
        print(f"[Success] Decrypted Message: {decrypted_message}")
    except ValueError as e:
        print(f"[Denied] Authentication Denied: {e}")

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TG-PFPKE Biometric Authentication System")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # enroll
    p_enroll = subparsers.add_parser("enroll", help="Enroll a fingerprint")
    p_enroll.add_argument("image", help="Path to fingerprint image (e.g. SOCOFing/Real/5__M_Left_index_finger.BMP)")

    # encrypt
    p_encrypt = subparsers.add_parser("encrypt", help="Encrypt a message for an enrolled user")
    p_encrypt.add_argument("user_id", help="Enrolled user ID (fingerprint filename without extension)")
    p_encrypt.add_argument("message", help="Secret message to encrypt")

    # decrypt
    p_decrypt = subparsers.add_parser("decrypt", help="Decrypt a message by authenticating with a fingerprint")
    p_decrypt.add_argument("user_id", help="Enrolled user ID (fingerprint filename without extension)")
    p_decrypt.add_argument("--engine", choices=["transformer", "gru", "iforest"], default="gru",
                           help="Risk evaluation engine (default: gru)")

    args = parser.parse_args()

    if args.command == "enroll":
        enroll(args)
    elif args.command == "encrypt":
        encrypt(args)
    elif args.command == "decrypt":
        decrypt(args)