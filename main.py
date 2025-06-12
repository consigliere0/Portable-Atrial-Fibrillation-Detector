# ====================
# IMPORTS
# ====================
from opensignalsreader import OpenSignalsReader
import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from wfdb import processing
import joblib
import os

# ========================
# LOAD DATA FROM BITALINO
# ========================
# replace path with local folder containing the .txt file from the BITalino
path = os.path.join(os.path.dirname(__file__), 'source', 'healthy_patient_example.txt')

acq = OpenSignalsReader(path)
signal_bitalino = acq.signal('ECG')
print(type(signal_bitalino))
print(signal_bitalino.shape)

# ======================================
# FILTERING AND QRS DETECTION FUNCTIONS
# ======================================
def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    """Applies a bandpass filter."""
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig)

def notch_filter(sig, fs, freq=50.0, Q=30):
    """Removes powerline noise (50 Hz notch)."""
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, sig)

def detect_qrs(sig, fs):
    """Detects R peaks using WFDB's GQRS."""
    return processing.gqrs_detect(sig, fs)

# ============================
# FEATURE EXTRACTION FUNCTION
# ============================
def extract_rr_parameters(qrs_inds, fs, min_beats=5):
    """
    Computes RR interval statistics for a window.
    Returns None if fewer than min_beats+1 peaks.
    """
    rr = np.diff(qrs_inds) / fs
    if len(rr) < min_beats:
        return None
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    nn50 = np.sum(np.abs(np.diff(rr)) > 0.05)
    pnn50 = nn50 / len(rr) * 100
    return {'meanRR': mean_rr, 'stdRR': std_rr, 'RMSSD': rmssd, 'pNN50': pnn50}

# ====================
# LOAD TRAINED MODEL
# ====================
model = joblib.load('models/rf_af_model.joblib')

# ======================================
# WINDOWING AND CLASSIFICATION FUNCTION
# ======================================
def load_windows(sig_f, qrs_inds, fs, window_sec=10):
    """
    Segments signal into windows and extracts RR features.
    
    Returns:
    - X: array (n,4) of features
    - y: array (n,) with labels (0 = normal, 1 = AF)
    """
    X_all, y_all = [], []
    win_len = int(window_sec * fs)
    n_wins = len(sig_f) // win_len
    valid_count = 0
    for w in range(n_wins):
        s = w * win_len
        e = s + win_len
        inds = qrs_inds[(qrs_inds >= s) & (qrs_inds < e)] - s
        feats = extract_rr_parameters(inds, fs)
        if feats is None:
            continue
        vals = [feats[k] for k in ('meanRR', 'stdRR', 'RMSSD', 'pNN50')]
        x = np.array(vals).reshape(1, -1)
        label = model.predict(x)[0]
        X_all.append(vals)
        y_all.append(label)
        valid_count += 1
        print(f"Processed: {valid_count}/{n_wins} valid windows")

    if not X_all:
        return np.empty((0, 4)), np.empty((0,))
    return np.array(X_all), np.array(y_all)

# ========================
# PROCESSING ECG SIGNAL
# ========================
# Downsample BITalino signal from 1000 Hz to 250 Hz
signal = signal_bitalino[::4]
print(signal.shape)

# Apply filters
signal_filtered = bandpass_filter(signal, 250)
signal_filtered = notch_filter(signal_filtered, 250)

# Detect R peaks
qrs_inds = detect_qrs(signal_filtered, 250)
print("Filtered signal and detected R peaks.")
print(qrs_inds.shape)

# Predict AF for each 10-second window
prediction = load_windows(signal_filtered, qrs_inds, 250, window_sec=10)
print(prediction)
# 0: no AF // 1: AF