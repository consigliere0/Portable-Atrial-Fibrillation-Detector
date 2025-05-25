print('Hello')

from pyhrv.hrv import hrv
from opensignalsreader import OpenSignalsReader
from biosppy.signals.ecg import ecg
import joblib

# IMPORT DATA
# Specify the file path
# From bitalino:
# path = "C:\Users\JULIANA BLANCO\Documents\OpenSignals (r)evolution\files\Prueba1.txt"
path = 'Prueba1.txt'
# Load the acquisition file
acq = OpenSignalsReader(path)

# Get the ECG signal
signal = acq.signal('ECG')
# What kind of data is this?
print(type(signal))

model = joblib.load("models/rf_af_model.joblib")