import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from wfdb import processing

def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig)

def notch_filter(sig, fs, freq=50.0, Q=30):
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, sig)

def detect_qrs(sig, fs):
    return processing.gqrs_detect(sig, fs)

def extract_rr_parameters(qrs_inds, fs):
    rr = np.diff(qrs_inds) / fs
    if len(rr) == 0:
        return {k: np.nan for k in ('meanRR', 'stdRR', 'RMSSD', 'pNN50')}
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    nn50 = np.sum(np.abs(np.diff(rr)) > 0.05)
    pnn50 = nn50 / len(rr) * 100
    return {'meanRR': mean_rr, 'stdRR': std_rr, 'RMSSD': rmssd, 'pNN50': pnn50}

def load_windows(record_name, db_path, window_sec=10, min_af_sec=0.0):
    sig_path = os.path.join(db_path, record_name)
    record = wfdb.rdrecord(sig_path)
    ann = wfdb.rdann(sig_path, 'atr')
    print(f"[{record_name}] Symbols arrel: {sorted(set(ann.symbol))}")
    print(f"[{record_name}] Aux notes úniques: {sorted(set(ann.aux_note))}")

    sig = record.p_signal[:, 0]
    fs = record.fs
    sig_f = notch_filter(bandpass_filter(sig, fs), fs)
    qrs_inds = detect_qrs(sig_f, fs)

    # Troba la primera aparició d'AFIB
    afib_samples = [ann.sample[i] for i in range(len(ann.aux_note)) if 'AFIB' in ann.aux_note[i]]
    first_afib_sample = min(afib_samples) if afib_samples else len(sig_f)

    samples_per_win = int(window_sec * fs)
    num_windows = len(sig_f) // samples_per_win
    X, y = [], []

    for w in range(num_windows):
        start = w * samples_per_win
        end = start + samples_per_win
        qrs_win = qrs_inds[(qrs_inds >= start) & (qrs_inds < end)] - start
        feats = extract_rr_parameters(qrs_win, fs)
        vec = [feats[k] for k in ('meanRR', 'stdRR', 'RMSSD', 'pNN50')]
        if any(np.isnan(vec)):
            continue

        # Etiquetem com a AFIB si la finestra inclou qualsevol anotació AFIB
        is_afib = any(
            start <= ann.sample[i] < end and 'AFIB' in ann.aux_note[i]
            for i in range(len(ann.sample))
        )
        # Si la finestra és abans del primer AFIB → forcem que és NO-AF
        if end <= first_afib_sample:
            label = 0
        else:
            label = 1 if is_afib else 0

        X.append(vec)
        y.append(label)

    return np.array(X), np.array(y)