
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:01:11 2026

@author: manue
"""

#ACA AGREGAMOS LOS NUEVOSMETODOS Q NO SON VENTANAS
#ACA EMPEZAMOS A MODIFICAR LA PLANTILLA PARA PASABANDAS

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio
from matplotlib.patches import Rectangle
#scipy.io (sio): leer y escribir archivos de datos, entre ellos los .mat de MATLAB.

plt.close('all')

#%% Lectura de ECG

fs_ecg = 1000  # Hz

# Cargar ECG con ruido
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_leadruido = mat_struct['ecg_lead'].flatten()
#.flatten() transforma la matriz en un vector unidimensional

# Graficar señal original COMPLETA
plt.figure(figsize=(12, 5))
plt.plot(ecg_one_leadruido)
plt.title('ECG completo con ruido')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid()
plt.show()

# Graficar un tramo de la señal original (zoom)
plt.figure(figsize=(10, 4))
plt.plot(ecg_one_leadruido[5000:7000])
plt.title('ECG con ruido - Zoom')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid()
plt.show()


#%%...................Diseño de filtro.................

wp1 = 1    #[Hz]
ws1 = 0.1  #[Hz]
wp2 = 35   #[Hz]
ws2 = 50   #[Hz]

Gpass = 3
Gstop = 40

ripple = Gpass
att = Gstop

sos_butt = sig.iirdesign(
    [wp1, wp2], [ws1, ws2], Gpass/2, Gstop/2,
    ftype='butter', fs=fs_ecg, output='sos'
)

sos_cheby2 = sig.iirdesign(
    [wp1, wp2], [ws1, ws2], Gpass/2, Gstop/2,
    ftype='cheby2', fs=fs_ecg, output='sos'
)

sos_cauer = sig.iirdesign(
    [wp1, wp2], [ws1, ws2], Gpass/2, Gstop/2,
    ftype='cauer', fs=fs_ecg, output='sos'
)

y_min, y_max = -100, 5



#FIR DESIGN

#si elijo tipo 2, el problema esla simetria xq quiero demora entera. siempre es preferible N impar para retardos enteros
#si quisiera tipo 2 hago filt filt

taps_win_hamm = 131
taps_win_box = 151
taps_win_hann = 2651
delta_w = 5

#los pares coordenados de la rta deseada. le pusimos una plantilla de pasabajos
#Ahora la respuesta deseada es pasabanda
# Filtro FIR por ventanas

# b_win_boxcar = sig.firwin2(
#     taps_win_box,
#     [0.0, ws1, wp1, wp2-2, wp2+delta_w, fs_ecg/2],
#     [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
#     window='boxcar', fs=fs_ecg
# )

# b_win_hamming = sig.firwin2(
#     taps_win_hamm,
#     [0.0, ws1, wp1, wp2+2, wp2+delta_w, fs_ecg/2],
#     [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
#     window='hamming', fs=fs_ecg
# )
ws1_modif=0.5

b_win_hann = sig.firwin2(
    taps_win_hann,
    [0.0, ws1_modif, wp1, wp2, ws2, fs_ecg/2],
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    window='hann', fs=fs_ecg
)


#%% FIR design Least Squares

taps_firls = 2251
wp2_diseno_ls = 43
ws2_diseno_ls = 45

b_firls = sig.firls(
    taps_firls,
    [0.0, ws1, wp1, wp2_diseno_ls, ws2_diseno_ls, fs_ecg/2],
    [0.0, 0.0, 0.98, 0.98, 0.0, 0.0],
    weight=[500.0, 5.0, 500.0],
    fs=fs_ecg
)

#%% FIR design Parks-McClellan

taps_remez = 1251

b_remez = sig.remez(
    taps_remez,
    [0.0, ws1, wp1, wp2_diseno_ls, ws2_diseno_ls, fs_ecg/2],
    [0.0, 0.9, 0.0],
    weight=[100.0, 10.0, 100.0],
    fs=fs_ecg
)

#%% Funciones de visualización

y_min, y_max = -100, 5

#con esto cambiamos el worn para poder repartir mejor los puntos

#constryuo una lista q de 0 a 1 meta 2mil puntos espaciados logaritmicamente
#log ayuda xq las tendencias tienden a cosas q son 1/omega.

ff = np.append(
    np.logspace(np.log10(0.001), np.log10(2), 2000),
    np.logspace(np.log10(2.01), np.log10(50), 2000)
)

ff = np.append(ff, np.linspace(50, fs_ecg/2, 1000, endpoint=True))

ff = np.unique(np.append(ff, [0, ws1, wp1, wp2, ws2]))
#en las trasniciones meinteresa mucha resolucion xq son transiciones abruptas. 
#fuera de la transicion no me interesa. 
#solo para el freqz, es para poder visualizar y ver si cumple o no cumple 

def calc_freq_sos(sos):
    # Respuesta en frecuencia de una pasada
    f, h = sig.freqz_sos(sos, worN=ff, fs=fs_ecg)

    # Respuesta efectiva de sosfiltfilt (dos pasadas)
    # Se cuadran los módulos y la fase resultante es cero
    h_doble = np.abs(h)**2

    return f, h_doble


def calc_freq_fir(bfir):
    # Respuesta en frecuencia en Hz, de 0 a Nyquist
    resp_freq = sig.freqz(bfir, 1, worN=ff, fs=fs_ecg)

    return resp_freq


def calc_resp(resp_freq, leg_str, fig_n_axes, ax_retardo):

    (f, h) = resp_freq

    # Módulo en dB y fase desenrollada en radianes
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    fase_rad = np.unwrap(np.angle(h))
    #np.angle puede devolver angulos con saltos. np.unwrap() corrige sumando o restando múltiplos de 2pi, permitiendo representar una fase continua.

    # Retardo de grupo: -d(fase)/d(omega), expresado en segundos
    omega = 2 * np.pi * f  # rad/s
    retardo_s = -np.gradient(fase_rad, omega)

    # En frecuencias donde el filtro prácticamente rechaza la señal,
    # la fase y el retardo pueden ser poco significativos.
    umbral_db = -80
    valido = modulo_db > umbral_db
    retardo_s_plot = np.where(valido, retardo_s, np.nan)

    (fig, (ax_mod, ax_fase)) = fig_n_axes

    # --- Módulo ---
    ax_mod.plot(f, modulo_db, linewidth=1.5,
                label=leg_str)
    ax_mod.set_ylim(y_min, y_max)
    ax_mod.set_ylabel("Módulo (dB)")
    ax_mod.set_title("Respuesta en frecuencia y plantilla")
    ax_mod.grid(True)

    # --- Fase ---
    ax_fase.plot(f, fase_rad, label=leg_str, linewidth=1.2)

    for borde in (ws1, wp1, wp2, ws2):
        ax_fase.axvline(borde, color="gray", linestyle=":")

    ax_fase.set_xlabel("Frecuencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.grid(True)

    # --- Retardo de grupo ---
    ax_retardo.plot(f, retardo_s_plot, label=leg_str)


#%% Graficar la respuesta de los filtros

fig_n_axes = plt.subplots(
    2, 1, sharex=True, figsize=(10, 7),
    constrained_layout=True
)

# Figura independiente para comparar retardos
fig_retardo, ax_retardo = plt.subplots(figsize=(10, 4))
# Respuestas de los filtros IIR
calc_resp(calc_freq_sos(sos_butt), 'Butterworth', fig_n_axes, ax_retardo)
calc_resp(calc_freq_sos(sos_cheby2), 'Chebyshev II', fig_n_axes, ax_retardo)
calc_resp(calc_freq_sos(sos_cauer), 'Cauer', fig_n_axes, ax_retardo)

# Respuestas de los filtros FIR
# calc_resp(calc_freq_fir(b_win_boxcar), 'FIR Boxcar', fig_n_axes, ax_retardo)
# calc_resp(calc_freq_fir(b_win_hamming), 'FIR Hamming', fig_n_axes, ax_retardo)
calc_resp(calc_freq_fir(b_win_hann), 'FIR Hann', fig_n_axes, ax_retardo)
calc_resp(calc_freq_fir(b_firls), 'FIR LS', fig_n_axes, ax_retardo)
calc_resp(calc_freq_fir(b_remez), 'FIR Parks-McClellan', fig_n_axes, ax_retardo)

# Ploteo de la plantilla

(fig, (ax_mod, ax_fase)) = fig_n_axes

# Banda pasante: zona no permitida por debajo de -ripple
rect_pasante = Rectangle(
    (wp1, y_min), wp2-wp1, -ripple-y_min,
    facecolor="red", alpha=0.12,
    edgecolor="red", hatch="xxx", linewidth=1.2,
    label='Plantilla'
)
ax_mod.add_patch(rect_pasante)

# Banda de rechazo: zona no permitida por encima de -att

# Rechazo inferior
rect_rechazo_inf = Rectangle(
    (0, -att), ws1, y_max+att,
    facecolor="red", alpha=0.12,
    edgecolor="red", hatch="xxx", linewidth=1.2
)
ax_mod.add_patch(rect_rechazo_inf)

# Rechazo superior
rect_rechazo = Rectangle(
    (ws2, -att), fs_ecg/2-ws2, y_max+att,
    facecolor="red", alpha=0.12,
    edgecolor="red", hatch="xxx", linewidth=1.2
)
ax_mod.add_patch(rect_rechazo)

# Límites de la plantilla
for borde in (ws1, wp1, wp2, ws2):
    ax_mod.axvline(borde, color="gray", linestyle=":")

# Etiquetas de zonas
ax_mod.text(
    (wp1+wp2)/2, y_max-6,
    "Banda pasante\npermitida: 0 a −3 dB",
    ha="center", va="top", fontsize=9
)

ax_mod.text(
    (ws2+100)/2, y_max-6,
    "Banda de rechazo\npermitida: ≤ −40 dB",
    ha="center", va="top", fontsize=9
)

ax_mod.text(
    (wp2+ws2)/2, y_min+5,
    "Transición",
    ha="center", va="bottom", fontsize=9
)

ax_mod.set_xlim(0, 100)
ax_mod.legend(loc="lower left", fontsize=8)
ax_fase.legend(loc="lower left", fontsize=8)


# --- Gráfico de retardo de grupo ---

ax_retardo.axvline(wp1, color="orange", linestyle="--",
                   label=f"wp1 = {wp1} Hz")

ax_retardo.axvline(wp2, color="orange", linestyle=":",
                   label=f"wp2 = {wp2} Hz")

ax_retardo.axvline(ws1, color="red", linestyle="--",
                   label=f"ws1 = {ws1} Hz")

ax_retardo.axvline(ws2, color="red", linestyle=":",
                   label=f"ws2 = {ws2} Hz")

ax_retardo.set_xlabel("Frecuencia (Hz)")
ax_retardo.set_ylabel("Retardo de grupo (s)")
ax_retardo.set_title("Retardo de grupo de los filtros IIR y FIR")
ax_retardo.set_xlim(0, 70)
# ax_retardo.set_ylim(0, 0.12)  # Zoom vertical
ax_retardo.grid(True)
ax_retardo.legend()

plt.show()



#%% Aplicar filtros al ECG

# IIR: filtrado hacia adelante y hacia atrás
ecg_butt = sig.sosfiltfilt(sos_butt, ecg_one_leadruido)
ecg_cheby2 = sig.sosfiltfilt(sos_cheby2, ecg_one_leadruido)
ecg_cauer = sig.sosfiltfilt(sos_cauer, ecg_one_leadruido)

# filtrado causal (una sola pasada)
# ecg_butt = sig.sosfilt(sos_butt, ecg_one_leadruido)
# ecg_cheby2 = sig.sosfilt(sos_cheby2, ecg_one_leadruido)
# ecg_cauer = sig.sosfilt(sos_cauer, ecg_one_leadruido)

# Aplicar filtros FIR al ECG
# ecg_win_boxcar = sig.lfilter(b_win_boxcar, 1, ecg_one_leadruido)
# ecg_win_hamming = sig.lfilter(b_win_hamming, 1, ecg_one_leadruido)
ecg_win_hann = sig.lfilter(b_win_hann, 1, ecg_one_leadruido)

# Aplicar FIRLS al ECG
ecg_firls = sig.lfilter(b_firls, 1, ecg_one_leadruido)

# Aplicar Parks-McClellan al ECG
ecg_remez = sig.lfilter(b_remez, 1, ecg_one_leadruido)

# Demora de grupo de los FIR: (N-1)/2 muestras
# Al comparar con el ECG original compensamos la demora al graficar
# demora_box = (b_win_boxcar.shape[0]-1)//2
# demora_hamm = (b_win_hamming.shape[0]-1)//2

demora_hann = (b_win_hann.shape[0]-1)//2
demora_firls = (b_firls.shape[0]-1)//2
demora_remez = (b_remez.shape[0]-1)//2

demora_max = max(demora_hann, demora_firls, demora_remez)

#%% Regiones de interés con ruido

cant_muestras = ecg_one_leadruido.shape[0]

regs_interes = (
    [4000, 5500],    # muestras
    [10000, 11000],  # muestras
)

for ii in regs_interes:

    # Intervalo limitado de 0 a cant_muestras
    inicio = max(0, int(ii[0]))
    fin = min(cant_muestras, int(ii[1]))

    if inicio >= fin:
        continue

    zoom_region = np.arange(inicio, fin, dtype=int)

    plt.figure(figsize=(12, 5))

    plt.plot(zoom_region, ecg_one_leadruido[zoom_region],
             label='ECG con ruido', linewidth=1.5, alpha=0.6)

    # Filtros IIR
    plt.plot(zoom_region, ecg_butt[zoom_region],
             label='Butterworth')

    plt.plot(zoom_region, ecg_cheby2[zoom_region],
             label='Chebyshev II')

    plt.plot(zoom_region, ecg_cauer[zoom_region],
             label='Cauer')

    # FIR alineados con el ECG original (demora compensada)
    zoom_fir = zoom_region[
        zoom_region + demora_max < cant_muestras
    ]

    # FIR Hann
    plt.plot(zoom_fir, ecg_win_hann[zoom_fir + demora_hann],
             label='FIR Hann')

    # FIR cuadrados mínimos
    plt.plot(zoom_fir, ecg_firls[zoom_fir + demora_firls],
             label='FIR LS')

    # FIR Parks-McClellan
    plt.plot(zoom_fir, ecg_remez[zoom_fir + demora_remez],
             label='FIR Parks-McClellan')

    plt.title('Región: ' + str(inicio) + ' a ' + str(fin))
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.grid()
    plt.legend()
    plt.show()

#%% Regiones de interés sin ruido

regs_interes = (
    np.array([5, 5.2]) * 60 * fs_ecg,    # minutos a muestras
    np.array([12, 12.4]) * 60 * fs_ecg,  # minutos a muestras
    np.array([15, 15.2]) * 60 * fs_ecg,  # minutos a muestras
)

for ii in regs_interes:

    # Intervalo limitado de 0 a cant_muestras
    inicio = max(0, int(ii[0]))
    fin = min(cant_muestras, int(ii[1]))

    if inicio >= fin:
        continue

    zoom_region = np.arange(inicio, fin, dtype=int)

    plt.figure(figsize=(12, 5))

    plt.plot(zoom_region, ecg_one_leadruido[zoom_region],
             label='ECG original', linewidth=1.5, alpha=0.6)

    # Filtros IIR
    plt.plot(zoom_region, ecg_butt[zoom_region],
             label='Butterworth')

    plt.plot(zoom_region, ecg_cheby2[zoom_region],
             label='Chebyshev II')

    plt.plot(zoom_region, ecg_cauer[zoom_region],
             label='Cauer')

    # FIR alineados con el ECG original (demora compensada)
    zoom_fir = zoom_region[
        zoom_region + demora_max < cant_muestras
    ]

    # FIR Hann
    plt.plot(zoom_fir, ecg_win_hann[zoom_fir + demora_hann],
             label='FIR Hann')

    # FIR cuadrados mínimos
    plt.plot(zoom_fir, ecg_firls[zoom_fir + demora_firls],
             label='FIR LS')

    # FIR Parks-McClellan
    plt.plot(zoom_fir, ecg_remez[zoom_fir + demora_remez],
             label='FIR Parks-McClellan')

    plt.title('Región: ' + str(inicio) + ' a ' + str(fin))
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.grid()
    plt.legend()
    plt.show()

# #%% Respuesta al impulso-CLASE

# delta_k = sig.unit_impulse(1000)

# resp_imp_butt = sig.sosfilt(sos_butt, delta_k)
# # resp_imp_cheby2 = sig.sosfilt(sos_cheby2, delta_k)
# # resp_imp_cauer = sig.sosfilt(sos_cauer, delta_k)

# plt.figure(figsize=(10, 4))

# plt.plot(resp_imp_butt, label='Butterworth')
# # plt.plot(resp_imp_cheby2, label='Chebyshev II')
# # plt.plot(resp_imp_cauer, label='Cauer')

# # Respuesta al impulso FIR: los coeficientes b son h[n]
# # plt.plot(b_win_boxcar, label='FIR Boxcar')
# # plt.plot(b_win_hamming, label='FIR Hamming')
# plt.plot(b_win_hann, label='FIR Hann')
# plt.plot(b_firls, label='FIR LS')
# plt.plot(b_remez, label='FIR Parks-McClellan')

# plt.xlabel('Tiempo (ms)')
# plt.ylabel('Amplitud')
# plt.title('Respuesta al impulso')
# plt.grid()
# plt.legend()
# plt.show()

#%% Respuesta al impulso - Comparación de filtros

# Impulso ubicado en el centro para evitar efectos de borde
cant_imp = 20000
centro_imp = cant_imp // 2

delta_k = np.zeros(cant_imp)
delta_k[centro_imp] = 1

# Respuesta al impulso IIR
# Mismo filtrado que aplicamos al ECG
resp_imp_butt = sig.sosfiltfilt(sos_butt, delta_k)
resp_imp_cheby2 = sig.sosfiltfilt(sos_cheby2, delta_k)
resp_imp_cauer = sig.sosfiltfilt(sos_cauer, delta_k)

# Eje temporal IIR, centrado en el impulso
t_iir = (np.arange(cant_imp) - centro_imp) / fs_ecg * 1000

# Respuesta al impulso FIR: los coeficientes b son h[n]
# Compensamos la demora para comparar los filtros alineados
t_hann = (np.arange(len(b_win_hann)) - demora_hann) / fs_ecg * 1000
t_firls = (np.arange(len(b_firls)) - demora_firls) / fs_ecg * 1000
t_remez = (np.arange(len(b_remez)) - demora_remez) / fs_ecg * 1000

plt.figure(figsize=(12, 5))

# IIR
plt.plot(t_iir, resp_imp_butt, label='Butterworth')
plt.plot(t_iir, resp_imp_cheby2, label='Chebyshev II')
plt.plot(t_iir, resp_imp_cauer, label='Cauer')

# FIR
plt.plot(t_hann, b_win_hann, label='FIR Hann')
plt.plot(t_firls, b_firls, label='FIR LS')
plt.plot(t_remez, b_remez, label='FIR Parks-McClellan')

plt.xlim(-2000, 2000)
plt.xlabel('Tiempo respecto del impulso (ms)')
plt.ylabel('Amplitud')
plt.title('Respuesta al impulso de los filtros')
plt.grid()
plt.legend()
plt.show()