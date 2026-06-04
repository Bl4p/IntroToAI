import hashlib
import numpy as np
import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.fernet import Fernet

class PFPKECrypto:
    def __init__(self):
        self.public_parameter = "TG-PFPKE-256-CURVE-PARAM"

    def key_gen(self, fuzzy_picture_data):
        x = fuzzy_picture_data['raw']
        np.random.seed(int(np.sum(x))) 
        sk = np.random.randn(*x.shape)
        
        c_sketch = x - sk
        c = {"sketch": c_sketch, "enrolled_x": x} 
        pk = hashlib.sha256(sk.tobytes() + self.public_parameter.encode()).hexdigest()
        
        return {"pk": pk, "c": c}, sk

    def _derive_symmetric_key(self, raw_matrix):
        key_material = hashlib.sha256(raw_matrix.tobytes()).digest()
        return base64.urlsafe_b64encode(key_material)

    def encrypt(self, pk_f, message):
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        svk_bytes = public_key.public_bytes_raw()
        
        sym_key = self._derive_symmetric_key(pk_f['c']['enrolled_x'])
        f = Fernet(sym_key)
        ciphertext_bytes = f.encrypt(message.encode())
        
        signature = private_key.sign(ciphertext_bytes)
        
        return {"svk": svk_bytes, "CT": ciphertext_bytes, "sigma": signature}

    def decrypt(self, pk_f, fuzzy_picture_data, ciphertext, alpha=1.0, t_base=25.0, tau_nu_base=0.15):
        x_prime = fuzzy_picture_data['raw']
        
        max_noise_allowed = tau_nu_base * alpha
        mean_noise = np.mean(fuzzy_picture_data['nu'])
        if mean_noise > max_noise_allowed:
            raise ValueError(f"Abort: Biometric noise ({mean_noise:.4f}) exceeds trust threshold ({max_noise_allowed:.4f}).")
            
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(ciphertext['svk'])
        try:
            public_key.verify(ciphertext['sigma'], ciphertext['CT'])
        except Exception:
            raise ValueError("Abort: Ciphertext signature validation failed! Data tampered.")
            
        t_active = t_base * alpha
        enrolled_x = pk_f['c']['enrolled_x']
        distance = np.mean(np.abs(x_prime - enrolled_x))
        
        if distance < t_active:
             sym_key = self._derive_symmetric_key(enrolled_x)
             f = Fernet(sym_key)
             try:
                 return f.decrypt(ciphertext['CT']).decode()
             except Exception:
                 raise ValueError("Decryption Failed: Key derivation mismatch.")
        else:
             raise ValueError(f"Decryption Failed: Distance ({distance:.2f}) exceeded physical threshold ({t_active:.2f}).")