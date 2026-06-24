from flask import Flask, render_template, jsonify, request, send_file
import os
import pickle
import glob
import numpy as np
import cv2
import sqlite3
import uuid
from biometric_scanner import BiometricScanner
from telemetry_db import TelemetryDB
from pfpke_crypto import PFPKECrypto

app = Flask(__name__)

KEYSTORE_DIR = "keystore"
FINGERPRINT_DIR = "SOCOFing/Real"
MAX_FINGERPRINTS = 100

# Cache risk engines so they aren't reloaded on every request
_engine_cache = {}

def _get_engine(engine_name):
    if engine_name not in _engine_cache:
        if engine_name == "gru":
            from tg_risk_gru import TGEvaluator_GRU
            _engine_cache[engine_name] = TGEvaluator_GRU()
        elif engine_name == "iforest":
            from tg_risk_iforest import TGEvaluator_IForest
            _engine_cache[engine_name] = TGEvaluator_IForest()
        else:
            from tg_risk_model import TGEvaluator
            _engine_cache[engine_name] = TGEvaluator()
    return _engine_cache[engine_name]

def _keystore_path(user_id):
    return os.path.join(KEYSTORE_DIR, f"{user_id}.pkl")

def _save_keystore(user_id, data):
    os.makedirs(KEYSTORE_DIR, exist_ok=True)
    with open(_keystore_path(user_id), "wb") as f:
        pickle.dump(data, f)

def _load_keystore(user_id):
    path = _keystore_path(user_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
        
        # MIGRATION: Convert legacy single-file keystores to the new messages array format
        if "messages" not in data:
            data["messages"] = []
            if "ciphertext" in data:
                data["messages"].append({
                    "id": str(uuid.uuid4()),
                    "title": data.get("title", "Untitled Document"),
                    "ciphertext": data["ciphertext"],
                    "content_length": data.get("content_length", 0)
                })
                # Clean up legacy fields
                if "ciphertext" in data: del data["ciphertext"]
                if "title" in data: del data["title"]
                if "content_length" in data: del data["content_length"]
                _save_keystore(user_id, data) # save migrated format
        return data

def _list_enrolled_users():
    if not os.path.exists(KEYSTORE_DIR):
        return []
    users = []
    for f in sorted(os.listdir(KEYSTORE_DIR)):
        if f.endswith(".pkl"):
            user_id = os.path.splitext(f)[0]
            ks = _load_keystore(user_id)
            if ks:
                users.append({
                    "user_id": user_id,
                    "public_key": ks.get("public_key", ""),
                    "messages": [
                        {"id": m["id"], "title": m["title"]} for m in ks.get("messages", [])
                    ]
                })
    return users

def _fuzzy_debug(fuzzy_data):
    return {
        "mu_mean": round(float(np.mean(fuzzy_data['mu'])), 6),
        "mu_std": round(float(np.std(fuzzy_data['mu'])), 6),
        "eta_mean": round(float(np.mean(fuzzy_data['eta'])), 6),
        "eta_std": round(float(np.std(fuzzy_data['eta'])), 6),
        "nu_mean": round(float(np.mean(fuzzy_data['nu'])), 6),
        "nu_std": round(float(np.std(fuzzy_data['nu'])), 6),
        "raw_mean": round(float(np.mean(fuzzy_data['raw'])), 4),
        "raw_std": round(float(np.std(fuzzy_data['raw'])), 4),
    }

def _telemetry_debug(t):
    return {
        "ip": t['ip'],
        "geo": [round(t['geo'][0], 4), round(t['geo'][1], 4)],
        "device_hash": t['device_hash'][:16] + "...",
        "time": t['time'],
    }

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/fingerprints')
def list_fingerprints():
    files = sorted(glob.glob(os.path.join(FINGERPRINT_DIR, "*.BMP")))[:MAX_FINGERPRINTS]
    names = [os.path.basename(f) for f in files]
    return jsonify({"fingerprints": names})

@app.route('/api/fingerprint-image/<filename>')
def get_fingerprint_image(filename):
    path = os.path.join(FINGERPRINT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype='image/bmp')

@app.route('/api/users')
def list_users():
    return jsonify({"users": _list_enrolled_users()})

@app.route('/api/enroll', methods=['POST'])
def enroll():
    data = request.json
    fingerprint_name = data.get("fingerprint")
    if not fingerprint_name:
        return jsonify({"status": "error", "message": "No fingerprint selected."}), 400

    user_id = os.path.splitext(fingerprint_name)[0]

    # Check if already enrolled
    existing = _load_keystore(user_id)
    if existing:
        return jsonify({
            "status": "info",
            "message": f"Already enrolled.",
            "debug": {
                "public_key": existing.get("public_key", ""),
                "user_id": user_id,
            }
        })

    # Load fingerprint
    image_path = os.path.join(FINGERPRINT_DIR, fingerprint_name)
    base_fp = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if base_fp is None:
        return jsonify({"status": "error", "message": f"Cannot read image: {fingerprint_name}"}), 400

    scanner = BiometricScanner()
    crypto = PFPKECrypto()
    db = TelemetryDB()

    # 10 scans with simulated variance
    scans = [scanner.rotate_upright(scanner.simulate_scan(base_fp, 0.04, 3)) for _ in range(10)]
    fuzzy_data = scanner.extract_fuzzy_picture(scans)

    pk_f, secret_key = crypto.key_gen(fuzzy_data)
    public_key_string = pk_f['pk']

    # Persist
    _save_keystore(user_id, {
        "user_id": user_id,
        "pk_f": pk_f,
        "sk": secret_key,
        "public_key": public_key_string,
        "image_path": image_path,
        "messages": [], # Initialize empty message vault
    })

    telemetry = db.capture_telemetry()
    db.log_attempt(public_key_string, telemetry, attempt_type="enrollment")

    return jsonify({
        "status": "success",
        "message": "Fingerprint enrolled successfully.",
        "debug": {
            "user_id": user_id,
            "public_key": public_key_string,
            "fuzzy_vectors": _fuzzy_debug(fuzzy_data),
            "telemetry": _telemetry_debug(telemetry),
            "scans_performed": 10,
        }
    })

@app.route('/api/encrypt', methods=['POST'])
def encrypt_route():
    data = request.json
    user_id = data.get("user_id")
    title = data.get("title", "Untitled Document")
    content = data.get("content", "")

    if not user_id or not content:
        return jsonify({"status": "error", "message": "User and content are required."}), 400

    keystore = _load_keystore(user_id)
    if keystore is None:
        return jsonify({"status": "error", "message": f"User '{user_id}' not enrolled."}), 404

    crypto = PFPKECrypto()
    ciphertext = crypto.encrypt(keystore['pk_f'], content)

    message_id = str(uuid.uuid4())
    keystore['messages'].append({
        "id": message_id,
        "title": title,
        "ciphertext": ciphertext,
        "content_length": len(content)
    })
    
    _save_keystore(user_id, keystore)

    return jsonify({
        "status": "success",
        "message": f"Message '{title}' encrypted successfully.",
        "debug": {
            "public_key": keystore['public_key'],
            "title": title,
            "message_id": message_id,
            "content_length": len(content),
            "ciphertext_bytes": len(ciphertext['CT']),
            "signature_present": ciphertext['sigma'] is not None,
        }
    })

@app.route('/api/decrypt', methods=['POST'])
def decrypt_route():
    data = request.json
    user_id = data.get("user_id")
    message_id = data.get("message_id")
    engine_name = data.get("engine", "gru")
    noise_override = data.get("noise_override", 0.06)

    if not user_id or not message_id:
        return jsonify({"status": "error", "message": "User ID and Message ID are required."}), 400

    keystore = _load_keystore(user_id)
    if keystore is None:
        return jsonify({"status": "error", "message": f"User '{user_id}' not enrolled."}), 404
    
    # Find the specific message
    message = next((m for m in keystore.get('messages', []) if m['id'] == message_id), None)
    if message is None:
        return jsonify({"status": "error", "message": "Encrypted message not found."}), 404

    pk_f = keystore['pk_f']
    ciphertext = message['ciphertext']
    public_key = keystore['public_key']
    title = message['title']

    # Load enrolled fingerprint and simulate a fresh re-scan
    base_fp = cv2.imread(keystore['image_path'], cv2.IMREAD_GRAYSCALE)
    if base_fp is None:
        return jsonify({"status": "error", "message": "Cannot read enrolled image."}), 500

    scanner = BiometricScanner()
    db = TelemetryDB()

    # Use the noise_override from the UI slider (or default 0.06)
    noise_variance = float(noise_override)
    login_scan = scanner.rotate_upright(
        scanner.simulate_scan(base_fp, noise_variance=noise_variance, blur_kernel_size=3)
    )
    login_fuzzy = scanner.extract_fuzzy_picture([login_scan])

    # Risk evaluation
    evaluator = _get_engine(engine_name)
    telemetry = db.capture_telemetry()
    
    # Telemetry override from UI
    if data.get('ip_override'): telemetry['ip'] = data['ip_override']
    if data.get('lat_override') and data.get('lon_override'): 
        telemetry['geo'] = (float(data['lat_override']), float(data['lon_override']))
    if data.get('device_hash_override'): telemetry['device_hash'] = data['device_hash_override']
    history = db.get_login_history(public_key)

    s_anomaly, alpha = evaluator.evaluate_risk(telemetry, history)

    debug = {
        "engine": engine_name,
        "public_key": public_key,
        "title": title,
        "noise_applied": noise_variance,
        "fuzzy_vectors": _fuzzy_debug(login_fuzzy),
        "risk": {
            "s_anomaly": round(s_anomaly, 4),
            "alpha": round(alpha, 4),
            "tau_nu_base": 0.30,
            "max_noise_allowed": round(0.30 * alpha, 4),
            "mean_biometric_noise": round(float(np.mean(login_fuzzy['nu'])), 4),
        },
        "telemetry": _telemetry_debug(telemetry),
        "history_count": len(history),
        "history": [
            {"ip": h['ip'], "geo": list(h['geo']),
             "device_hash": h['device_hash'][:16] + "...", "time": h['time']}
            for h in history[:10]
        ],
    }

    # Hard gate
    if s_anomaly > 0.95:
        debug["decrypted_text"] = None
        debug["denial_reason"] = "CRITICAL: Anomaly score exceeded 0.95 kill-switch threshold."
        return jsonify({"status": "denied", "message": "CRITICAL THREAT DETECTED.", "debug": debug})

    db.log_attempt(public_key, telemetry, attempt_type="login")

    # Cryptographic decryption
    crypto = PFPKECrypto()
    try:
        decrypted = crypto.decrypt(pk_f, login_fuzzy, ciphertext, alpha=alpha)
        debug["decrypted_text"] = decrypted
        return jsonify({"status": "success", "message": "Authentication and Decryption successful!", "decrypted_content": decrypted, "debug": debug})
    except ValueError as e:
        debug["decrypted_text"] = None
        debug["denial_reason"] = str(e)
        return jsonify({"status": "denied", "message": str(e), "debug": debug})

@app.route('/api/database/<table>')
def database_explorer(table):
    if table not in ("enrollments", "login_attempts"):
        return jsonify({"status": "error", "message": "Invalid table."}), 400

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 15))
    search = request.args.get("search", "").strip()
    offset = (page - 1) * per_page

    conn = sqlite3.connect("tg_pfpke.db")
    cur = conn.cursor()

    where_clause = ""
    params = []
    
    if search:
        # Search across ip, public_key, and device_hash
        where_clause = "WHERE ip_address LIKE ? OR public_key LIKE ? OR device_hash LIKE ?"
        like_search = f"%{search}%"
        params = [like_search, like_search, like_search]

    # Get total count with search filter
    cur.execute(f"SELECT COUNT(*) FROM {table} {where_clause}", params)
    total = cur.fetchone()[0]

    # Get rows
    query = f"""
        SELECT id, public_key, latitude, longitude, ip_address, device_hash, timestamp 
        FROM {table} 
        {where_clause}
        ORDER BY id DESC LIMIT ? OFFSET ?
    """
    cur.execute(query, params + [per_page, offset])
    rows = cur.fetchall()
    conn.close()

    return jsonify({
        "table": table,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "search_term": search,
        "rows": [
            {
                "id": r[0],
                "public_key": (r[1][:16] + "...") if r[1] else "",
                "latitude": round(r[2], 4) if r[2] else 0,
                "longitude": round(r[3], 4) if r[3] else 0,
                "ip_address": r[4] or "",
                "device_hash": (r[5][:16] + "...") if r[5] else "",
                "timestamp": r[6] or 0,
            }
            for r in rows
        ]
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
