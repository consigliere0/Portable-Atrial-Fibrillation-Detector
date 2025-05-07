'''
    DEFINICIÓ PIPELINE
Cadena d'etapes de processament que s'apliquen sempre en el m ateix ordre. Es transformen les dades fins arribar a un resultat.
Enlloc de fer-ho manualment s'empra una pipeline per automatitzar-ho, fer el codi eficient, i evitar errors. Estalvia a més codi repetitiu.
És un objecte de la llibreria "scikit-learn"

    CONTINGUTS FITXER
Conté una pipeline bàsica (s'haurà d'anar perfilant) per a la detecció d'AF a partir de la MIT-BIH database
Inclou:
    - Filtrat del senyal (bandpass + notch)
    - Detecció de pics QRS
    - Extracció de paràmetres dels intervals RR
    - Càrrega i preparació de finestres (dividir la mostra en seccions de 30s)
    - Entrenament del model RandomForest
    - Avaluació del model
'''

import os                                            # comunicar amb el sistema operatiu del PC
import numpy as np
import wfdb                                          # llegir registres de EGC en format PhysioNet (WFDB)
from scipy.signal import butter, filtfilt, iirnotch  # aplicar filtres digitals al senyal
from sklearn.pipeline import Pipeline                # combinar pre-processat i model en una sola estructura
from sklearn.preprocessing import StandardScaler     # escalar les dades abans d'entrenar
from sklearn.ensemble import RandomForestClassifier  # algorisme d'aprenentatge automàtic per classificar
from sklearn.model_selection import train_test_split, cross_val_score #separar dades i avaluar models
from wfdb import processing


# ---------------
# Filtrat digital
# ---------------
def bandpass_filter (sig, fs, lowcut = 0.5, highcut = 40.0, order = 4):
    nyq = fs/2
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype = 'band')
    return filtfilt(b, a, sig)

def notch_filter (sig, fs, freq = 50.0, Q = 30):
    b, a = iirnotch(freq/(fs/2),Q )
    return filtfilt(b, a, sig)

# ---------------
# Detecció QRS
# ---------------
def detect_qrs(sig, fs):
    qrs_inds = wfdb.processing.gqrs_detect(sig, fs)
    return qrs_inds

# ---------------
# Paràmetres
# ---------------
def extract_rr_parameters(qrs_inds, fs):
    rr = np.diff(qrs_inds) / fs                          # intervals en segons
    mean_rr = np.mean(rr)                                # mitjana RR
    std_rr = np.std(rr)                                  # desviació estàndard
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))     # Índex de variabilitat del ritme
    nn50 = np.sum(np.abs(np.diff(rr)) > 0.05)
    pnn50 = nn50 / len(rr) * 100                         # pNN50

    return {'meanRR': mean_rr, 'stdRR': std_rr, 'RMSSD': rmssd, 'pNN50': pnn50}

# -----------------------------------
# Càrrega i preparació de finestres
# -----------------------------------

def load_windows(record_name, db_path, window_sec=30):
    record_path = os.path.join(db_path, record_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')        # llegeix les anotacions mèdiques 'ann' x saber quan hi ha AF
    sig = record.p_signal[:,0]                  # carrega el senyal
    fs = record.fs                              # carrega la freqüència de mostreig

    # Filtrar el senyal
    sig_f = bandpass_filter(sig, fs)
    sig_f = notch_filter(sig_f, fs)

    # Detectar pics QRS
    qrs_inds = detect_qrs(sig_f, fs)

    # Preparar finestres
    samples_per_win = window_sec * fs
    num_windows = int(len(sig_f) / samples_per_win) # dividim el senyal en finestres de 30s
    X = []
    y = []
    
    for w in range(num_windows):
        # start = int(len(sig_f) / samples_per_win)
        start = int(w * samples_per_win)
        end = int(start + samples_per_win)

        # QRS dins finestra
        qrs_win = qrs_inds[(qrs_inds >= start) & (qrs_inds < end)] - start
        feats = extract_rr_parameters(qrs_win, fs)
        X.append([feats['meanRR'], feats['stdRR'], feats['RMSSD'], feats['pNN50']])

        # Etiqueta: >50% anotacions AFIB dins finestra?
        ann_win = [a for a in ann.sample if start <= a < end]
        # af_count = sum([1 for a in ann_win if ann.symbol[ann.sample.index(a)] == 'AFIB'])
        af_count = sum([1 for i, a in enumerate(ann.sample) if start <= a < end and ann.symbol[i] == 'AFIB'])

        label = 1 if (af_count / len(ann_win) if ann_win else 0) > 0.5 else 0
        y.append(label)
    return np.array(X), np.array(y)


#-----------------------
# Entrenament del model
# ----------------------
def train_model(X,y):
    clf = Pipeline([
        ('Scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    clf.fit(X,y)
    return clf


# --------------------
# Avaluació del model
# --------------------
def evaluate_model(X, y):
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    print(f'AUC mitjana (5-fold): {scores.mean():.3f}')

# ------------------
# Avaluació creuada
# ------------------
def evaluate_model(X,y):
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    print(f'AUC mitjana (5-fold): {scores.mean():.3f}')


# --------
# Exemple
# --------
if __name__ == '__main__':
    # Exemple d'ús amb un registre de la carpeta 'afdb'
    DATA_DIR = 'afdb'
    rec = '04043'
    rec_path = os.path.join(DATA_DIR, rec)
    X, y = load_windows(rec_path)
    print(f'Features shape: {X.shape}, Labels balance: {np.bincount(y)}')
    evaluate_model(X, y)


'''

CÓDIGO JULIANA

# Vamos a probar con una de las muestras de la base de datos disponibles
# Ruta al registro
record_path = '../data/Long-Term_AF/00'
# Leer la señal y anotaciones
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

# Mostrar información básica
print("Señal:", record.p_signal.shape)
print("Frecuencia de muestreo:", record.fs)
print("Número de anotaciones:", len(annotation.sample))
#print("Tipos de eventos:", annotation.symbol)

# Aplicamos los filtros a la señal
filtered_signal= bandpass_filter (record.p_signal[0:2000, 0] #recortamos la señal en las pruebas para ganar velocidad
                                  , record.fs, lowcut = 0.5, highcut = 40.0, order = 4)
filtered_signal = notch_filter(filtered_signal,record.fs,50,30)

# Detecció QRS
# La opción de Laura no me funciona, pq no reconoce "processing" dentro de wfdb
# qrs_inds = wfdb.processing.gqrs_detect(filtered_signal, record.fs)
# Le pedimos una alternativa al chatgpt

from biosppy.signals import ecg

out = ecg.ecg(signal=filtered_signal, sampling_rate=record.fs, show=False)
qrs_inds = out['rpeaks']

# Extracció de paràmetres

RR_parameters=extract_rr_parameters(qrs_inds, record.fs)
print(RR_parameters['pNN50'])

# Las anotaciones que tenemos solo nos indican cuando hay cambios de ritmo.
# Si hacemos la extracción de los parámetros QRS de tohdo,
#  aunque haya una parte con FA, el resto la compensará y, en promedio, el paciente "estará sano".
# USAR LAS ESTIQUETAS COMO CORTE? QUE CALCULE QRS DE CADA SEGMENTO?
# ENTRENAR CON LOS SEGMENTOS EL MACHINE LEARNING, ETIQUETADOS CON SI TIENEN O NO FA = RITMO POR ENCIMA DE X?
'''