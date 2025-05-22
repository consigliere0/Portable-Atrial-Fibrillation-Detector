# pipeline_prova.py
import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from wfdb import processing


def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    """Aplica un filtre pas‐banda."""
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig)


def notch_filter(sig, fs, freq=50.0, Q=30):
    """Elimina soroll de xarxa (notch 50 Hz)."""
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, sig)


def detect_qrs(sig, fs):
    """Detecció de pics R amb GQRS de WFDB."""
    return processing.gqrs_detect(sig, fs)


def extract_rr_parameters(qrs_inds, fs, min_beats=5):
    """
    Calcula estadístiques RR de la finestra.
    Retorna None si hi ha menys de min_beats+1 pics.
    """
    # intervals RR en segons
    rr = np.diff(qrs_inds) / fs
    if len(rr) < min_beats:
        return None
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    nn50 = np.sum(np.abs(np.diff(rr)) > 0.05)
    pnn50 = nn50 / len(rr) * 100
    return {'meanRR': mean_rr, 'stdRR': std_rr, 'RMSSD': rmssd, 'pNN50': pnn50}


def load_windows(record_names, db_path, window_sec=10):
    """
    Carrega finestres i extreu característiques RR per múltiples registres.

    Params:
    - record_names: llista de codis (sense extensió)
    - db_path: directori WFDB
    - window_sec: durada finestra en s

    Retorna:
    - X: array (n,4)
    - y: array (n,) 0=sinusal, 1=AF
    """
    X_all, y_all = [], []
    for rec in record_names:
        base = os.path.join(db_path, rec)
        # llegir registre i anotacions
        record = wfdb.rdrecord(base)
        # per normals no hi ha anotacions 'AFIB', usem ann.symbol per evitar error
        try:
            ann = wfdb.rdann(base, 'atr')
        except Exception:
            ann = None

        sig = record.p_signal[:, 0]
        fs = record.fs
        # filtratge
        sig_f = notch_filter(bandpass_filter(sig, fs), fs)
        qrs_inds = detect_qrs(sig_f, fs)

        # defineix límit per AF: primer sample amb aux_note 'AFIB'
        first_af = len(sig_f)
        if ann:
            af_samples = [ann.sample[i] for i, txt in enumerate(ann.aux_note) if 'AFIB' in txt]
            if af_samples:
                first_af = min(af_samples)

        # segmentació en finestres
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
            vals = [feats[k] for k in ('meanRR','stdRR','RMSSD','pNN50')]
            # etiqueta AF si la finestra cau després del primer AF i hi ha anotació
            label = 0
            if ann and e > first_af:
                # marca si hi ha algun 'AFIB' dins la finestra
                label = int(any(s <= samp < e and 'AFIB' in ann.aux_note[i]
                                for i, samp in enumerate(ann.sample)))
            X_all.append(vals)
            y_all.append(label)
            valid_count += 1
        print(f"Processed {rec}: {valid_count}/{n_wins} finestres vàlides")

    if not X_all:
        return np.empty((0,4)), np.empty((0,))
    return np.array(X_all), np.array(y_all)