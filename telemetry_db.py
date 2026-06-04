import sqlite3
import time
import hashlib
import requests
import platform
import subprocess

class TelemetryDB:
    def __init__(self, db_name="tg_pfpke.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        for table in ["enrollments", "login_attempts"]:
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    public_key TEXT, latitude REAL, longitude REAL,
                    ip_address TEXT, device_hash TEXT, timestamp INTEGER
                )
            ''')
        self.conn.commit()

    def _get_hardware_uuid(self):
        sys_os = platform.system()
        try:
            if sys_os == "Darwin":
                output = subprocess.check_output(['system_profiler', 'SPHardwareDataType']).decode('utf-8')
                for line in output.split('\n'):
                    if 'Hardware UUID' in line:
                        return line.split(':')[1].strip()
            elif sys_os == "Linux":
                return subprocess.check_output(['cat', '/sys/class/dmi/id/product_uuid']).decode('utf-8').strip()
        except Exception:
            pass
        return "Unknown_Hardware_Fallback"

    def capture_telemetry(self):
        try:
            resp = requests.get('http://ip-api.com/json/', timeout=3).json()
            public_ip = resp.get('query', '127.0.0.1')
            lat, lon = resp.get('lat', 0.0), resp.get('lon', 0.0)
        except requests.RequestException:
            public_ip, lat, lon = "127.0.0.1", 0.0, 0.0

        hardware_uuid = self._get_hardware_uuid()

        return {
            "geo": (lat, lon), "ip": public_ip, 
            "device_hash": hashlib.sha256(hardware_uuid.encode()).hexdigest(),
            "time": int(time.time())
        }

    def log_attempt(self, pk, telemetry, attempt_type="enrollment"):
        table = "enrollments" if attempt_type == "enrollment" else "login_attempts"
        self.cursor.execute(f'''
            INSERT INTO {table} (public_key, latitude, longitude, ip_address, device_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (pk, telemetry['geo'][0], telemetry['geo'][1], telemetry['ip'], telemetry['device_hash'], telemetry['time']))
        self.conn.commit()

    def get_login_history(self, pk, limit=10):
        self.cursor.execute('''
            SELECT latitude, longitude, ip_address, device_hash, timestamp 
            FROM login_attempts WHERE public_key = ? ORDER BY timestamp DESC LIMIT ?
        ''', (pk, limit))
        return [{"geo": (r[0], r[1]), "ip": r[2], "device_hash": r[3], "time": r[4]} for r in self.cursor.fetchall()]