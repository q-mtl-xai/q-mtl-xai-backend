# preprocessing.py
# Funciones de preprocesamiento

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d


def apply_notch(signal, fs, notch_freq=50.0, quality=30.0):
    b, a = iirnotch(notch_freq, quality, fs)
    return filtfilt(b, a, signal)


def apply_bandpass(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, signal)


def filter_ecg(signal, fs, notch_freq=50.0):
    signal = apply_notch(signal, fs, notch_freq)
    signal = apply_bandpass(signal, fs, 0.5, 40.0, order=4)
    return signal


def resample_signal(signal, fs_orig, fs_target):
    if fs_orig == fs_target:
        return signal
    n_orig = len(signal)
    n_target = int(n_orig * fs_target / fs_orig)
    t_orig = np.linspace(0, 1, n_orig)
    t_target = np.linspace(0, 1, n_target)
    f = interp1d(t_orig, signal, kind="cubic", fill_value="extrapolate")
    return f(t_target)


def normalize_minmax(signal):
    smin, smax = np.min(signal), np.max(signal)
    if smax - smin < 1e-8:
        return np.zeros_like(signal)
    return (signal - smin) / (smax - smin)


def preprocess_ecg(signal, fs_orig, fs_target=250, window_size=250):
    signal = np.asarray(signal, dtype=np.float32)

    # 1D (latido o ventana)
    if signal.ndim == 1:
        signal = filter_ecg(signal, fs_orig)
        signal = resample_signal(signal, fs_orig, fs_target)
        if len(signal) > window_size:
            signal = signal[:window_size]
        elif len(signal) < window_size:
            signal = np.pad(signal, (0, window_size - len(signal)), mode='edge')
        signal = normalize_minmax(signal)
        return signal

    # 2D (12 leads, isquemia)
    else:
        n_leads, n_samples = signal.shape
        processed = []
        for lead in range(n_leads):
            s = signal[lead]
            # Detectar si ya viene normalizada [0, 1]
            already_normalized = (s.min() >= -0.01 and s.max() <= 1.01)
            if already_normalized:
                # Solo recortar o padear
                if len(s) > window_size:
                    s = s[:window_size]
                elif len(s) < window_size:
                    s = np.pad(s, (0, window_size - len(s)), mode='edge')
            else:
                # Cruda: filtro + resample + min-max
                s = filter_ecg(s, fs_orig)
                s = resample_signal(s, fs_orig, fs_target)
                if len(s) > window_size:
                    s = s[:window_size]
                elif len(s) < window_size:
                    s = np.pad(s, (0, window_size - len(s)), mode='edge')
                s = normalize_minmax(s)
            processed.append(s)
        return np.array(processed, dtype=np.float32)
