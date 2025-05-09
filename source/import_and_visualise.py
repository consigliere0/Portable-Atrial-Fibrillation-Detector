import wfdb
import matplotlib.pyplot as plt
import numpy as np
#ldkdkds
# Ruta al registro sin extensión
record_path = '../data/Long-Term_AF/05'
# record_path = '../data/MIT-BIH_afdb/04126'

'''
# Comprobar que los archivos funcionen correctamente
info = wfdb.rdheader(record_path)
print("Frecuencia de muestreo:", info.fs)
print("Duración (segundos):", info.sig_len / info.fs)
print("Muestras totales:", info.sig_len)
'''

# Leer la señal y anotaciones
record = wfdb.rdrecord(record_path)
# record.p_signal = Array NumPy [n_muestras x n_canales] con la señal digital convertida a milivoltios
annotation = wfdb.rdann(record_path, 'atr')
# annotation.sample: Lista de índices (en muestras) donde ocurren los eventos (como R-peaks)
# annotation.symbol: Lista de símbolos ASCII con la anotación de cada latido
senyal = record.p_signal


# Mostrar información básica
print("Señal:", record.p_signal.shape)
print("Frecuencia de muestreo:", record.fs)
print("Número de anotaciones:", len(annotation.sample))
print(annotation.sample)


eventos =[]
evento_indices = [] # indice en la señal del electrocardiograma en el que sucede el evento
normal_indices = [] # indice en la señal del electrocardiograma en el que sucede el latido normal
# Buscamos los latidos no normales y otros eventos
for indice, item in enumerate(annotation.symbol): # Revisar todas las anotaciones
    if item != 'N': # Si el latido es anormal
        eventos.append(item) # Apuntar el tipo de evento
        muestra_del_evento = annotation.sample[indice]
        evento_indices.append(muestra_del_evento)
        # Suele haber mínimo unas doscientas muestras entre evento y evento
    '''
    else:
        muestra_del_evento = annotation.sample[indice]
        normal_indices.append(muestra_del_evento)
'''

print('Número de eventos: ', len(eventos))
print("Tipos de eventos:", eventos)
print("Índices de los eventos: ", evento_indices)


# Graficar la señal con anotaciones
plt.figure(figsize=(12, 4))
plt.plot(record.p_signal[:, 0], label='ECG canal 1')
# Marca los latidos anormales y otras anotaciones
plt.plot(evento_indices, record.p_signal[evento_indices, 0], 'ro', label='Eventos')
# Vamos a marcar las zonas de latidos normales
for item in evento_indices:
    x_fill = np.arange(item - 100, item + 100)
    plt.fill_between(x_fill, -4, 4, color='red', alpha=0.2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Long-Term con anotaciones')
plt.legend()
plt.show()
