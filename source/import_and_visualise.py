import wfdb
import matplotlib.pyplot as plt

# Ruta al registro sin extensión
record_path = '../data/Long-Term_AF/00'

# Comprobar que los archivos funcionen correctamente
info = wfdb.rdheader(record_path)
print("Frecuencia de muestreo:", info.fs)
print("Duración (segundos):", info.sig_len / info.fs)
print("Muestras totales:", info.sig_len)


# Leer la señal y anotaciones
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')
senyal = record.p_signal


# Mostrar información básica
print("Señal:", record.p_signal.shape)
print("Frecuencia de muestreo:", record.fs)
print("Número de anotaciones:", len(annotation.sample))


eventos =[]
# Buscamos los latidos no normales y otros eventos
for item in annotation.symbol:
    if item != 'N':
        eventos.append(item)

print('Número de eventos: ', len(eventos))
print("Tipos de eventos:", eventos)


# Graficar la señal con anotaciones
plt.figure(figsize=(12, 4))
plt.plot(record.p_signal[:, 0], label='ECG canal 1')
plt.plot(annotation.sample, record.p_signal[annotation.sample, 0], 'ro', label='Anotaciones')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Long-Term con anotaciones')
plt.legend()
plt.show()
