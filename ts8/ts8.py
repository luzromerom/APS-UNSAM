# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:00:55 2026

@author: manue
"""

#%% FILTRO DE MEDIANA

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio
from scipy.interpolate import CubicSpline

plt.close('all')

fs_ecg = 1000  # Hz

mat_struct = sio.loadmat('ECG_TP4.mat')
ecg_one_leadruido = mat_struct['ecg_lead'].flatten()

cant_muestras = len(ecg_one_leadruido)


#%% ESTIMACIÓN DE LA LÍNEA DE BASE - MEDIANA
# sig.medfilt(señal, kernel_size)



# Primera mediana de 200 ms, agrego 1 muestra mas xq medfilt toma n impar y es centrada
b_200 = sig.medfilt(ecg_one_leadruido, 201)
# Primera mediana: intenta eliminar los QRS, que son eventos relativamente cortos.

# Segunda mediana de 600 ms
b_mediana = sig.medfilt(b_200, 601)
# intenta eliminar estructuras más anchas, especialmente las ondas T.

# ECG corregido
ecg_mediana = ecg_one_leadruido - b_mediana

# La mediana elimina bien los QRS y las ondas T, pero puede generar
# escalones o cambios bruscos de pendiente en la línea de base estimada.
# Esto pasa porque elige el valor central de la ventana, no promedia.
# Al restar esa estimación podemos meter deformaciones en el ECG,
# incluso en zonas que estaban limpias, especialmente cerca del QRS.
# Otra mediana no necesariamente corrige esos escalones.

# #%% FILTROS LINEALES DE LA TS7 PARA COMPARAR-- clase, no parte de ts8

# # Mismos parámetros 

# wp1 = 1
# ws1 = 0.1
# wp2 = 35
# ws2 = 50

# Gpass = 3
# Gstop = 40

# # IIR Butterworth
# sos_butt = sig.iirdesign(
#     [wp1, wp2], [ws1, ws2], Gpass/2, Gstop/2,
#     ftype='butter', fs=fs_ecg, output='sos'
# )

# ecg_butt = sig.sosfiltfilt(sos_butt, ecg_one_leadruido)

# # FIR Hann
# taps_win_hann = 2651
# ws1_modif = 0.5

# b_win_hann = sig.firwin2(
#     taps_win_hann,
#     [0.0, ws1_modif, wp1, wp2, ws2, fs_ecg/2],
#     [0.0, 0.0, 0.995, 0.995, 0.0, 0.0],
#     window='hann', fs=fs_ecg
# )

# ecg_win_hann = sig.lfilter(b_win_hann, 1, ecg_one_leadruido)

# # Compensación del retardo del FIR
# demora_hann = (taps_win_hann - 1)//2


# #%% REGIONES DE INTERÉS - COMPARACIÓN DE FILTROS

# regs_interes = (
#     [4000, 5500],       # Región relativamente limpia
#     [10000, 11000],     # Región utilizada en TS7
#     [720000, 744000],   # Mucho movimiento de línea de base
#     [900000, 912000],   # Otra región con interferencia
# )

# for inicio, fin in regs_interes:

#     fin = min(fin, cant_muestras)

#     if inicio >= fin:
#         continue

#     zoom_region = np.arange(inicio, fin)

#     plt.figure(figsize=(12, 5))

#     plt.plot(zoom_region, ecg_one_leadruido[zoom_region],
#              label='ECG original', alpha=0.6)

#     plt.plot(zoom_region, ecg_butt[zoom_region],
#              label='Butterworth')

#     # FIR alineado con el ECG original
#     zoom_fir = zoom_region[zoom_region + demora_hann < cant_muestras]

#     plt.plot(zoom_fir, ecg_win_hann[zoom_fir + demora_hann],
#              label='FIR Hann')

#     plt.plot(zoom_region, ecg_mediana[zoom_region],
#              label='Mediana', color='red')

#     plt.title('Comparación de filtros: ' + str(inicio) + ' a ' + str(fin))
#     plt.xlabel('Muestras (#)')
#     plt.ylabel('Amplitud')
#     plt.grid()
#     plt.legend()
#     plt.show()


# #%% ESTIMACIÓN DE LA LÍNEA DE BASE

# # Visualizamos las dos medianas y el ECG original

# regs_interes = (
#     [4000, 5500],
#     [720000, 744000],
# )

# for inicio, fin in regs_interes:

#     zoom_region = np.arange(inicio, fin)

#     plt.figure(figsize=(12, 5))

#     plt.plot(zoom_region, ecg_one_leadruido[zoom_region],
#              label='ECG original', color='gray', alpha=0.6)

#     plt.plot(zoom_region, b_200[zoom_region],
#              label='Primera mediana - 201 muestras')

#     plt.plot(zoom_region, b_mediana[zoom_region],
#              label='Línea de base estimada - 601 muestras',
#              color='orange', linewidth=2)

#     plt.title('Estimación de la línea de base')
#     plt.xlabel('Muestras (#)')
#     plt.ylabel('Amplitud')
#     plt.grid()
#     plt.legend()
#     plt.show()

#%% SPLINES CÚBICOS - PUNTOS PQ


qrd_det = mat_struct['qrs_detections'].flatten().astype(int)

start_win = qrd_det - 60
end_win = qrd_det - 40

# para cada QRS tomamos una ventana de 20 muestras, desde 60 hasta 40 ms antes de su detección
#Se toman 20 muestras por latido porque, a 1000 Hz, esa ventana equivale a 20 ms
# un período completo de una interferencia sinusoidal de 50 Hz. para reducir esa interferencia antes de interpolar.

# Evitar ventanas fuera del registro
validos = (start_win >= 0) & (end_win <= cant_muestras)
qrd_det = qrd_det[validos]
start_win = start_win[validos]
end_win = end_win[validos]

# Tiempo medio de cada ventana
pq_t_hat = np.array([np.mean([a,b]) for a,b in zip(start_win, end_win)])

# Amplitud media del ECG original
pq_hat = np.array([np.mean(ecg_one_leadruido[a:b])
                   for a,b in zip(start_win, end_win)])

# Para visualizar los puntos sobre el ECG filtrado
pq_mediana = np.array([np.mean(ecg_mediana[a:b])
                       for a,b in zip(start_win, end_win)])





# #%% VISUALIZACIÓN DE LOS PUNTOS PQ


# plt.figure(figsize=(14, 5))

# plt.plot(ecg_mediana, label='ECG mediana', linewidth=0.5)

# plt.plot(qrd_det, ecg_mediana[qrd_det],
#          'xr', label='QRS', markersize=4)

# plt.plot(pq_t_hat, pq_mediana,
#          'og', label='Puntos PQ', markersize=3)

# plt.title('Localización de los segmentos PQ')
# plt.xlabel('Muestras (#)')
# plt.ylabel('Amplitud')
# plt.grid()
# plt.legend()
# plt.show()


# # zoom

# regs_interes = (
#     [92000, 95000],      # Región con puntos PQ bien ubicados
#     [720000, 723000],    # Región donde la elección es más dudosa
# )

# for inicio, fin in regs_interes:

#     zoom_region = np.arange(inicio, fin)

#     # QRS y puntos PQ dentro de la región
#     qrs_region = qrd_det[(qrd_det >= inicio) & (qrd_det < fin)]
#     pq_region = (pq_t_hat >= inicio) & (pq_t_hat < fin)

#     plt.figure(figsize=(12, 5))

#     plt.plot(zoom_region, ecg_mediana[zoom_region],
#              label='ECG mediana', color='blue')

#     # Detecciones QRS y ventanas PQ
#     for q in qrs_region:
#         plt.axvline(q, color='red', linestyle='--', alpha=0.5)
#         plt.axvspan(q-60, q-40, color='green', alpha=0.2)

#     # Promedio de cada ventana
#     plt.plot(pq_t_hat[pq_region], pq_mediana[pq_region],
#              'og', label='Promedio PQ')

#     plt.title('Verificación PQ: ' + str(inicio) + ' a ' + str(fin))
#     plt.xlabel('Muestras (#)')
#     plt.ylabel('Amplitud')
#     plt.grid()
#     plt.legend()
#     plt.show()



#%% INTERPOLACIÓN SPLINE CÚBICA

spline = CubicSpline(pq_t_hat, pq_hat)

n = np.arange(cant_muestras)

# Línea de base estimada
b_spline = spline(np.clip(n, pq_t_hat[0], pq_t_hat[-1]))#np.clip es x seguridad

# ECG corregido
ecg_spline = ecg_one_leadruido - b_spline


#%% COMPARACIÓN MEDIANA VS SPLINE

plt.figure(figsize=(12, 5))

plt.plot(b_mediana,
         label='Línea de base - Mediana')

plt.plot(b_spline,
         label='Línea de base - Spline')

plt.title('Comparación de estimaciones de línea de base')
plt.xlabel('Muestras (#)')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.show()

#%% zooms

regs_interes = (
    [4000, 5500],
    [720000, 744000],
    [900000, 912000],
)

for inicio, fin in regs_interes:

    zoom_region = np.arange(inicio, fin)

    plt.figure(figsize=(12,5))

    plt.plot(zoom_region, b_mediana[zoom_region],
             label='Línea de base - Mediana')

    plt.plot(zoom_region, b_spline[zoom_region],
             label='Línea de base - Spline')

    plt.title('Comparación de estimaciones de línea de base')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.legend()
    plt.show()


#%% 3) FILTRO ADAPTADO-busca encontrar partes dl ecg q se parecen a un patron qrs conocido

# patrón QRS conocido
qrs_p1 = mat_struct['qrs_pattern1'].flatten()

# Visualización del patrón
plt.figure(figsize=(10, 4))
plt.plot(qrs_p1, label='Patrón original')
plt.plot(qrs_p1[::-1], label='Patrón invertido')
plt.grid()
plt.legend()
plt.show()

#La correlación (usando sigfilter) consiste en desplazar el patrón sobre la señal, multiplicar las muestras que 
#coinciden y sumar esos productos.

#sigfilter usa convolución
#Para obtener una correlación mediante convolución, tenemos que invertir temporalmente el patrón.qrs_p1[::-1]

# filtro adaptado aplicado al ECG corregido cn spline
#sig.lfilter(b, a, señal),a=1 p/q sea fir
qrs_matched = sig.lfilter(qrs_p1[::-1], 1, ecg_spline)
#losb busco invertidos

# normalizo p/comparar las amplitudes señal-val.medio/desv estandar
ecg_norm = (ecg_spline - np.mean(ecg_spline)) / np.std(ecg_spline)

matched_norm = (qrs_matched - np.mean(qrs_matched)) / np.std(qrs_matched)

#%% VERIFICAR RETARDO DEL FILTRO ADAPTADO

plt.figure(figsize=(12, 5))

plt.plot(ecg_norm, label='ECG corregido')
plt.plot(matched_norm, label='Filtro adaptado')

plt.xlim(92000, 95000)

plt.xlabel('Muestras (#)')
plt.ylabel('Amplitud normalizada')
plt.grid()
plt.legend()
plt.show()

# Retardo medido entre los máximos del ECG y del filtro adaptado
retardo = 69  # muestras de distancia entre picos medida

plt.figure(figsize=(12, 5))

plt.plot(ecg_norm[:-retardo], label='ECG corregido')
plt.plot(matched_norm[retardo:], label='Filtro adaptado')

plt.xlim(92000, 95000)

plt.title('Salida del filtro adaptado - retardo compensado')
plt.xlabel('Muestras (#)')
plt.ylabel('Amplitud normalizada')
plt.grid()
plt.legend()
plt.show()




# #%%Diseño de filtro pasabajos
# wp = 23#[Hz]
# ws = 25 #[Hz]
# Gpass = 3
# Gstop = 20


# sos_butt_det = sig.iirdesign(
#     wp, ws, Gpass/2, Gstop/2,
#     ftype='butter', fs=fs_ecg, output='sos'
# )


# # saco el modulo
# matched_abs = np.abs(matched_norm)

# envolvente = sig.sosfiltfilt(sos_butt_det, matched_abs)

# # Compenso el retardo, elimino las primeras 69m
# envolvente = envolvente[retardo:]



# #%% COMPARACIÓN SOSFILT Y SOSFILTFILT

# inicio = 92000
# fin = 95000

# # Aplicamos una sola vez
# prueba_una = sig.sosfilt(sos_butt_det, matched_abs)

# # Aplicamos dos veces
# prueba_doble = sig.sosfiltfilt(sos_butt_det, matched_abs)

# plt.figure(figsize=(12, 5))

# plt.plot(np.arange(inicio, fin), matched_abs[inicio:fin],
#          label='Sin pasabajos', alpha=0.4)

# plt.plot(np.arange(inicio, fin), prueba_una[inicio:fin],
#          label='sosfilt')

# plt.plot(np.arange(inicio, fin), prueba_doble[inicio:fin],
#          label='sosfiltfilt')

# plt.xlim(inicio, fin)
# plt.grid()
# plt.legend()
# plt.show()

#%% FILTRO DE ORDEN - PERCENTIL 90

# Módulo de la salida del filtro adaptado
matched_abs = np.abs(matched_norm)

# Ventana de 41 muestras y percentil 90
ventana = 41
rank = 36

# Aplicamos el filtro de orden
envolvente = sig.order_filter(matched_abs, np.ones(ventana), rank)

# Compensamos el retardo del filtro adaptado
envolvente = envolvente[retardo:]


#%% COMPARACIÓN ANTES Y DESPUÉS DEL PASABAJOS

inicio = 92000
fin = 95000

plt.figure(figsize=(12, 5))

plt.plot(np.arange(inicio, fin),
         matched_abs[inicio+retardo:fin+retardo],
         label='Módulo sin suavizar', alpha=0.6)

plt.plot(np.arange(inicio, fin),
         envolvente[inicio:fin],
         label='Después del pasabajos', linewidth=2)

plt.title('Efecto del filtro pasabajos')
plt.xlabel('Muestras (#)')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.show()

#%% DETECCIÓN DE LATIDOS

# busco máximos de la envolvente cn find_peaks
#distance: cuanto deen estar separados 2 picos
#prominence dice cuanto sobresale un pico respecto de los valles q lo rodean
 
picos, propiedades = sig.find_peaks( envolvente, distance=400,height=0.5)

plt.figure(figsize=(12, 5))

plt.plot(envolvente, label='Señal de detección')

plt.plot(picos, envolvente[picos],
         'rx', label='Latidos detectados')

plt.title('Detección de QRS mediante filtro adaptado')
plt.xlabel('Muestras (#)')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.show()

print('Latidos detectados:', len(picos))



#%% VERIFICACIÓN DEL DETECTOR

inicio = 794800
fin = 798500

zoom = np.arange(inicio, fin)

# Detecciones dentro de la región
qrs_region = qrd_det[(qrd_det >= inicio) & (qrd_det < fin)]
picos_region = picos[(picos >= inicio) & (picos < fin)]

plt.figure(figsize=(14, 5))

# ECG original corregido
plt.plot(zoom, ecg_norm[zoom],
         label='ECG corregido', alpha=0.7)

# Envolvente del filtro adaptado
plt.plot(zoom, envolvente[zoom],
         label='Envolvente', color='orange', linewidth=2)

# Picos encontrados con find_peaks
plt.plot(picos_region, envolvente[picos_region],
         'rx', label='Picos detectados', markersize=8)

# QRS reales: líneas verticales verdes
for q in qrs_region:
    plt.axvline(q, color='green', linestyle='--', alpha=0.6)

plt.title('Verificación de detecciones QRS')
plt.xlabel('Muestras (#)')
plt.ylabel('Amplitud normalizada')
plt.grid()
plt.legend()
plt.show()


#%% 3b)

qrs_ref = np.sort(mat_struct['qrs_detections'].flatten())
picos = np.sort(picos)

tolerancia = 100  

TP = 0
i = 0
j = 0

# Comparamos los picos detectados con los QRS de referencia
while i < len(picos) and j < len(qrs_ref):

    if abs(picos[i] - qrs_ref[j]) <= tolerancia:
        TP += 1
        i += 1
        j += 1
    elif picos[i] < qrs_ref[j]:
        i += 1
    else:
        j += 1

FP = len(picos) - TP
FN = len(qrs_ref) - TP

sensibilidad = 100 * TP / len(qrs_ref)
VPP = 100 * TP / len(picos)

print('MATRIZ DE CONFUSIÓN')
print('                 Detectado   No detectado')
print(f'QRS real         {TP:8d}   {FN:8d}')
print(f'No QRS           {FP:8d}          -')

print(f'\nSensibilidad: {sensibilidad:.2f}%')
print(f'VPP: {VPP:.2f}%')



plt.figure(figsize=(12, 5))

plt.plot(ecg_spline, label='ECG sin línea de base')
plt.plot(picos, ecg_spline[picos], 'rx', label='Mis detecciones')
plt.plot(qrs_ref, ecg_spline[qrs_ref], 'go', label='QRS de referencia')

#plt.xlim(794800, 798500)
plt.legend()
plt.grid()
plt.show()