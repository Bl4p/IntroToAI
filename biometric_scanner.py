import numpy as np
import cv2

class BiometricScanner:
    def __init__(self):
        pass

    def simulate_scan(self, base_fingerprint, noise_variance=0.05, blur_kernel_size=3):
        ksize = int(blur_kernel_size) if int(blur_kernel_size) % 2 != 0 else int(blur_kernel_size) + 1
        blurred_scan = cv2.GaussianBlur(base_fingerprint, (ksize, ksize), 0) if ksize > 1 else base_fingerprint.copy()
        noise = np.random.normal(0, 255 * noise_variance, blurred_scan.shape)
        return np.clip(blurred_scan.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def rotate_upright(self, image):
        moments = cv2.moments(image)
        if moments['mu20'] != moments['mu02']:
            angle = 0.5 * np.arctan(2 * moments['mu11'] / (moments['mu20'] - moments['mu02']))
            angle_degrees = np.degrees(angle)
        else:
            angle_degrees = 0
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    def extract_fuzzy_picture(self, images):
        raw_vector = np.mean(images, axis=0)
        
        x_min, x_max = np.min(raw_vector), np.max(raw_vector)
        b = np.mean(raw_vector)
        a = (b + x_min) / 2.0
        c = (b + x_max) / 2.0

        mu_A = np.zeros_like(raw_vector)
        eta_A = np.zeros_like(raw_vector)
        nu_A = np.zeros_like(raw_vector)

        # L-R and Triangular Formulas
        mask_mu = (raw_vector > b) & (raw_vector <= c)
        mu_A[mask_mu] = (raw_vector[mask_mu] - b) / (c - b)
        
        mask_nu = (raw_vector >= a) & (raw_vector < b)
        nu_A[mask_nu] = (b - raw_vector[mask_nu]) / (b - a)

        mask_eta1 = (raw_vector >= b) & (raw_vector < c)
        mask_eta2 = (raw_vector >= a) & (raw_vector < b)
        eta_A[mask_eta1] = (c - raw_vector[mask_eta1]) / (c - b)
        eta_A[mask_eta2] = (raw_vector[mask_eta2] - a) / (b - a)

        normalization = mu_A + eta_A + nu_A + 1e-6
        scale_factor = np.where(normalization >= 1.0, 0.99 / normalization, 1.0)
        
        return {
            "mu": mu_A * scale_factor, 
            "eta": eta_A * scale_factor, 
            "nu": nu_A * scale_factor, 
            "raw": raw_vector
        }