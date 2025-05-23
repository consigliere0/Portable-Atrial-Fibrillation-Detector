print('Hello')

#from pyhrv.hrv import hrv
#from opensignalsreader import OpenSignalsReader
#from biosspy.signals.ecg import ecg

# IMPORT DATA
# Specify the file path
path = "C:\Users\JULIANA BLANCO\Documents\OpenSignals (r)evolution\files\Prueba1.txt"

# Load the acquisition file
acq = OpenSignalsReader(path)

# Get the ECG signal
signal = acq.signal('ECG')
# What kind of data is this?
print(type(signal))