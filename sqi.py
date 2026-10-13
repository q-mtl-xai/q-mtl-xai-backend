# sqi.py
# Funciones del Signal Quality Index
# Usa modelos entrenados (c3_sqi_ecg.pkl, c3_sqi_ppg.pkl)

import os
import numpy as np
import joblib
from scipy.stats import entropy as scipy_entropy


# === Cargar modelos SQI ===
SQI_DIR = os.path.dirname(__file__)
SQI_ECG_PATH = os.path.join(SQI_DIR, 'models', 'c3_sqi_ecg.pkl')
SQI_PPG_PATH = os.path.join(SQI_DIR, 'models', 'c3_sqi_ppg.pkl')

print("Cargando modelos SQI...")
sqi_ecg_data = joblib.load(SQI_ECG_PATH)
sqi_ppg_data = joblib.load(SQI_PPG_PATH)
print("Modelos SQI cargados")


def extract_sqi_features(signal):
    """Extrae las 8 features de calidad de una señal 1D."""
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    var = np.var(signal)
    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist[hist > 0]
    ent = scipy_entropy(hist)
    p95 = np.percentile(signal, 95)
    p5 = np.percentile(signal, 5)
    signal_power = (p95 - p5) ** 2
    noise_power = np.var(signal - np.median(signal))
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    amp = signal.max() - signal.min()
    zc = np.sum(np.diff(np.sign(signal)) != 0) / n
    energy = np.sum(signal ** 2) / n
    mean = np.mean(signal)
    std = np.std(signal) + 1e-10
    kurt = np.mean(((signal - mean) / std) ** 4)
    skew = np.mean(((signal - mean) / std) ** 3)
    return np.array([var, ent, snr, amp, zc, energy, kurt, skew], dtype=np.float32)


def predict_sqi_ecg(signal):
    """Predice SQI usando el modelo ECG entrenado."""
    features = extract_sqi_features(signal).reshape(1, -1)
    features_scaled = sqi_ecg_data['scaler'].transform(features)
    pred = sqi_ecg_data['model'].predict(features_scaled)[0]
    return str(pred)


def predict_sqi_ppg(signal):
    """Predice SQI usando el modelo PPG entrenado."""
    features = extract_sqi_features(signal).reshape(1, -1)
    features_scaled = sqi_ppg_data['scaler'].transform(features)
    pred = sqi_ppg_data['model'].predict(features_scaled)[0]
    return str(pred)


def simple_sqi(signal):
    """SQI simple basado en reglas (fallback)."""
    signal = np.asarray(signal, dtype=np.float64)
    var = np.var(signal)
    p95 = np.percentile(signal, 95)
    p5 = np.percentile(signal, 5)
    snr = 10 * np.log10((p95 - p5) ** 2 / (np.var(signal - np.median(signal)) + 1e-10))
    if snr > 15 and var > 0.001:
        return 'Good'
    elif snr > 5 and var > 0.0001:
        return 'Fair'
    else:
        return 'Poor'
