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
    - 
'''

import os                                            # comunicar amb el sistema operatiu del PC
import numpy as np
import wfdb                                          # llegir registres de EGC en format PhysioNet (WFDB)
from scipy.signal import butter, filtfilt, iirnotch  # aplicar filtres digitals al senyal
from sklearn.pipeline import Pipeline                # combinar pre-processat i model en una sola estructura
from sklearn.preprocessing import StandardScaler     # escalar les dades abans d'entrenar
from sklearn.ensemble import RandomForestClassifier  # algorisme d'aprenentatge automàtic per classificar
from sklearn.model_selection import train_test_split, cross_val_predict #separar dades i avaluar models


# ---------------
# Filtrat digital
# ---------------
def bandpass_filter (sig, fs, lowcut = 0.5, highcut = 40.0, order = 4):
    nyq = fs/2
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype = 'band')
    return filtfilt(b, a, sig) # la sortida és el senyal filtrat

def notch_filter (sig, fs, freq = 50.0, Q = 30):
    b, a = iirnotch(freq/(fs/2),Q )
    return filtfilt(b, a, sig) # la sortida és el senyal filtrat

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

    return {'rr': rr,'meanRR': mean_rr, 'stdRR': std_rr, 'RMSSD': rmssd, 'pNN50': pnn50} # la salida es un diccionario



# Vamos a probar con una de las muestras de la base de datos disponibles
# Ruta al registro
record_path = '../data/MIT-BIH_afdb/05091'
# Leer la señal y anotaciones
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

# Mostrar información básica
print("Señal:", record.p_signal.shape)
print("Frecuencia de muestreo:", record.fs)
print("Número de anotaciones:", len(annotation.sample))
print("Tipos de eventos:", annotation.symbol)

# Aplicamos los filtros a la señal
filtered_signal= bandpass_filter (record.p_signal[:, 0], record.fs, lowcut = 0.5, highcut = 40.0, order = 4)
filtered_signal = notch_filter(filtered_signal,record.fs,50,30)

# Detecció QRS
qrs_inds = wfdb.processing.gqrs_detect(filtered_signal, record.fs)

# Extracció de paràmetres
extract_rr_parameters(qrs_inds, record.fs)