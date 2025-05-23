# Importem els mòduls necessaris
# Per importar les dades preses al bitalino
from pyhrv.hrv import hrv
from opensignalsreader import OpenSignalsReader
from biosppy.signals.ecg import ecg
# Per processar la senyal
import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from wfdb import processing
# Per emprar el model 
import joblib

# IMPORTAR DADES DEL BITALINO
# Specify the file path
# From bitalino: (PROVAR RUTA EN ORDINADOR LAURA)
# path = "C:\Users\JULIANA BLANCO\Documents\OpenSignals (r)evolution\files\Prueba1.txt"
path = 'Prueba1.txt' #Aquest és l'arxiu de prova que tenim aquí
# Load the acquisition file
acq = OpenSignalsReader(path)

# Get the ECG signal
signal_bitalino = acq.signal('ECG')
# What kind of data is this?
print(type(signal_bitalino)) # <class 'numpy.ndarray'>
print(signal_bitalino.shape) # (11700, 1)
# FUNCIONS DE FILTARTGE I EXTRACCIÓ DE CARACTERÍSTIQUES
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


def load_windows(sig_f, qrs_inds, fs, window_sec=5): # Reduit a 5 segons pq la mostra es petita

    """
    Carrega finestres i extreu característiques RR per múltiples registres.

    Params:
    - sig_f: senyal filtrada
    - fs: freqüència de mostreig
    - window_sec: durada finestra en s

    Retorna:
    - X: array (n,4)
    - y: array (n,) 0=sinusal, 1=AF NO INTERESSA EN AQUEST CAS
    """
    X_all, y_all = [], []
    # segmentació en finestres
    win_len = int(window_sec * fs) # longitud finestra en samples
    n_wins = len(sig_f) // win_len # nombre de finestres
    valid_count = 0
    for w in range(n_wins): # per cada finestra
        # defineix límits de la finestra
        s = w * win_len
        e = s + win_len
        # selecciona pics dins la finestra
        inds = qrs_inds[(qrs_inds >= s) & (qrs_inds < e)] - s
        # calcula els paràmetres RR
        feats = extract_rr_parameters(inds, fs)
        if feats is None: # si no hi ha prou pics, salta
            continue
        vals = [feats[k] for k in ('meanRR','stdRR','RMSSD','pNN50')] # registra las característiques de la finestra


        # etiqueta AF si el model ho indica
        label = 0 # preset: normal
        # Ajustar les dades a les compatibles amb el model
        x = np.array(vals).reshape(1, -1)
        """
        if ann and e > first_af:
            # marca si hi ha algun 'AFIB' dins la finestra
            label = int(any(s <= samp < e and 'AFIB' in ann.aux_note[i]
                                for i, samp in enumerate(ann.sample)))
        """
        X_all.append(vals) # X_all és la llista de les característiques de totes les finestres
        y_all.append(label) # y_all és la llista de les etiquetes de totes les finestres
        valid_count += 1
        print(f"Processed: {valid_count}/{n_wins} finestres vàlides")

    if not X_all:
        return np.empty((0,4)), np.empty((0,))
    return np.array(X_all), np.array(y_all)


# El bitalino mostreja a 1000 Hz
# El registre de la Laura mostreja a 250 Hz
# Cal mostrejar el senyal de lbitalino per a que les freqüencies coincideixin
signal = signal_bitalino[::4]
print(signal.shape) # (2925, 1)
# S'apliquen els filtres
signal_filtered = bandpass_filter(signal, 250)
signal_filtered = notch_filter(signal_filtered, 250)

# Detecció de pics R
qrs_inds = detect_qrs(signal_filtered, 250)
print("Senyal filtrada i índexs de R detectats.")
print(qrs_inds.shape) # (14, 1)
# El codi ha de segmentar el senyal en finestres de 10 segons i aplicar a cadascuna el model per determinar si hi ha o no AF
aplicar_load_windows = load_windows(signal_filtered,qrs_inds, 250, window_sec=10)
print(aplicar_load_windows) 