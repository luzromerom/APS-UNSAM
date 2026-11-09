# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 21:28:05 2026

@author: manue
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:01:11 2026

@author: manue
"""
#ACA AGREGAMOS LOS NUEVOSMETODOS Q NO SON VENTANAS
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

wp = 35 #[Hz]
ws = 50 #[Hz]
Gpass = 3
Gstop = 40

ripple = Gpass
att = Gstop

sos_butt = sig.iirdesign(
    wp, ws, Gpass/2, Gstop/2,
    ftype='butter', fs=fs_ecg, output='sos'
)

sos_cheby2 = sig.iirdesign(
    wp, ws, Gpass/2, Gstop/2,
    ftype='cheby2', fs=fs_ecg, output='sos'
)

sos_cauer = sig.iirdesign(
    wp, ws, Gpass/2, Gstop/2,
    ftype='cauer', fs=fs_ecg, output='sos'
)

y_min, y_max = -100, 5

#FIR DESIGN

#si elijo tipo 2, el problema esla simetria xq quiero demora entera. siempre es preferible N impar para retardos enteros
#si quisiera tipo 2 hago filt filt

taps_win_hamm = 131
taps_win_box = 151
taps_win_hann=131
delta_w = 5


#los pares coordenados de la rta deseada. le pusimos una plantilla de pasabajos
# Filtro FIR por ventanas
b_win_boxcar = sig.firwin2(
    taps_win_box,
    [0.0, wp-2, wp+delta_w, fs_ecg/2],
    [1.0, 1.0, 0.0, 0.0],
    window='boxcar', fs=fs_ecg
)

b_win_hamming = sig.firwin2(
    taps_win_hamm,
    [0.0, wp+2, wp+delta_w, fs_ecg/2],
    [1.0, 1.0, 0.0, 0.0],
    window='hamming', fs=fs_ecg
)

b_win_hann = sig.firwin2(
    taps_win_hann,
    [0.0, wp, wp+delta_w, fs_ecg/2],
    [1.0, 1.0, 0.0, 0.0],
    window='hann', fs=fs_ecg
)

#%% FIR design Least Squares

taps_firls = 131

b_firls = sig.firls(
    taps_firls,
    [0.0, wp+6, ws+7, fs_ecg/2],
    [1.0, 1.0, 0.0, 0.0],
    weight=[1.0, 1.0],
    fs=fs_ecg
)

y_min, y_max = -100, 5  
  
def calc_freq_sos(sos):
   # Respuesta en frecuencia en Hz, de 0 a Nyquist
   resp_freq = sig.freqz_sos(sos, worN=8192, fs=fs_ecg)
   
   return resp_freq

def calc_freq_fir (bfir):
   # Respuesta en frecuencia en Hz, de 0 a Nyquist
   resp_freq = sig.freqz(bfir,1, worN=8192, fs=fs_ecg)
   
   return resp_freq

def calc_resp(resp_freq, leg_str, fig_n_axes, ax_retardo):

    (f,h)=resp_freq

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
    ax_fase.axvline(wp, color="gray", linestyle=":")
    ax_fase.axvline(ws, color="gray", linestyle=":")
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
calc_resp(calc_freq_sos(sos_butt), 'butter', fig_n_axes, ax_retardo)
# calc_resp(calc_freq_sos(sos_cheby2), 'cheby2', fig_n_axes, ax_retardo)
# calc_resp(calc_freq_sos(sos_cauer), 'cauer', fig_n_axes, ax_retardo)

# Respuestas de los filtros FIR
# calc_resp(calc_freq_fir(b_win_boxcar), 'FIR Boxcar', fig_n_axes, ax_retardo)
# calc_resp(calc_freq_fir(b_win_hamming), 'FIR Hamming', fig_n_axes, ax_retardo)
calc_resp(calc_freq_fir(b_win_hann), 'FIR Hann', fig_n_axes, ax_retardo)
calc_resp(calc_freq_fir(b_firls), 'FIR LS', fig_n_axes, ax_retardo)
# Ploteo de la plantilla

(fig, (ax_mod, ax_fase)) = fig_n_axes

# Banda pasante: zona no permitida por debajo de -ripple
rect_pasante = Rectangle(
    (0, y_min), wp, -ripple - y_min,
    facecolor="red", alpha=0.12,
    edgecolor="red", hatch="xxx", linewidth=1.2,
    label='Plantilla'
)
ax_mod.add_patch(rect_pasante)

# Banda de rechazo: zona no permitida por encima de -att
rect_rechazo = Rectangle(
    (ws, -att), fs_ecg / 2 - ws, y_max + att,
    facecolor="red", alpha=0.12,
    edgecolor="red", hatch="xxx", linewidth=1.2
)
ax_mod.add_patch(rect_rechazo)

# Límites de la plantilla
ax_mod.axvline(wp, color="gray", linestyle=":")
ax_mod.axvline(ws, color="gray", linestyle=":")

# Etiquetas de zonas
ax_mod.text(
    wp / 2, y_max - 6,
    "Banda pasante\npermitida: 0 a −3 dB",
    ha="center", va="top", fontsize=9
)

ax_mod.text(
    (ws + 100) / 2, y_max - 6,
    "Banda de rechazo\npermitida: ≤ −40 dB",
    ha="center", va="top", fontsize=9
)

ax_mod.text(
    (wp + ws) / 2, y_min + 5,
    "Transición",
    ha="center", va="bottom", fontsize=9
)

# Zoom para observar mejor la plantilla
ax_mod.set_xlim(0, 100)
ax_mod.legend(loc="lower left", fontsize=8)
ax_fase.legend(loc="lower left", fontsize=8)


# --- Gráfico de retardo de grupo ---
ax_retardo.axvline(wp, color="orange", linestyle="--",
                   label=f"wp = {wp} Hz")
ax_retardo.axvline(ws, color="red", linestyle="--",
                   label=f"ws = {ws} Hz")

ax_retardo.set_xlabel("Frecuencia (Hz)")
ax_retardo.set_ylabel("Retardo de grupo (s)")
ax_retardo.set_title("Retardo de grupo de los filtros IIR")
ax_retardo.set_xlim(0, 70)
ax_retardo.set_ylim(0, 0.12)  # Zoom vertical
ax_retardo.grid(True)
ax_retardo.legend()

plt.show()


#%% Aplicar filtros al ECG

ecg_butt = sig.sosfiltfilt(sos_butt, ecg_one_leadruido)
ecg_cheby2 = sig.sosfiltfilt(sos_cheby2, ecg_one_leadruido)
ecg_cauer = sig.sosfiltfilt(sos_cauer, ecg_one_leadruido)
# ecg_butt = sig.sosfilt(sos_butt, ecg_one_leadruido)
# ecg_cheby2 = sig.sosfilt(sos_cheby2, ecg_one_leadruido)
# ecg_cauer = sig.sosfilt(sos_cauer, ecg_one_leadruido)

# Aplicar filtros FIR al ECG
ecg_win_boxcar = sig.lfilter(b_win_boxcar, 1, ecg_one_leadruido)
ecg_win_hamming = sig.lfilter(b_win_hamming, 1, ecg_one_leadruido)
ecg_win_hann = sig.lfilter(b_win_hann, 1, ecg_one_leadruido)

# Aplicar FIRLS al ECG
ecg_firls = sig.lfilter(b_firls, 1, ecg_one_leadruido)

# Demora de grupo de los FIR: (N-1)/2 muestras
# Al comparar con el ECG original compensamos la demora al graficar
demora_box = (b_win_boxcar.shape[0]-1)//2
demora_hamm = (b_win_hamming.shape[0]-1)//2
demora_hann = (b_win_hann.shape[0]-1)//2
demora_firls = (b_firls.shape[0]-1)//2

demora_max = max(demora_box, demora_hamm, demora_hann, demora_firls)

# # Vector de tiempo
# t = np.arange(len(ecg_one_leadruido)) / fs_ecg

# # Región de interés
# inicio = 5000
# fin = 7000

# plt.figure(figsize=(12, 6))

# plt.plot(t[inicio:fin], ecg_one_leadruido[inicio:fin],
#          label='ECG con ruido', alpha=0.5)

# plt.plot(t[inicio:fin], ecg_butt[inicio:fin],
#          label='Butterworth')

# plt.plot(t[inicio:fin], ecg_cheby2[inicio:fin],
#          label='Chebyshev II')

# plt.plot(t[inicio:fin], ecg_cauer[inicio:fin],
#          label='Cauer')

# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.title('Comparación de filtros IIR sobre ECG')
# plt.grid()
# plt.legend()
# plt.show()

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

    # plt.plot(zoom_region, ecg_butt[zoom_region],
    #          label='Butterworth')

    # plt.plot(zoom_region, ecg_cheby2[zoom_region],
    #          label='Chebyshev II')

    # plt.plot(zoom_region, ecg_cauer[zoom_region],
    #          label='Cauer')

    # # FIR alineados con el ECG original (demora compensada)
    # zoom_fir = zoom_region[zoom_region + demora_max < cant_muestras]

    # plt.plot(zoom_fir, ecg_win_boxcar[zoom_fir + demora_box],
    #          label='FIR Boxcar')

    # plt.plot(zoom_fir, ecg_win_hamming[zoom_fir + demora_hamm],
    #          label='FIR Hamming')

    # plt.plot(zoom_fir, ecg_win_hann[zoom_fir + demora_hann],
             # label='FIR Hann')
    
    # IIR Butterworth
    plt.plot(zoom_region, ecg_butt[zoom_region],
         label='Butterworth')

# FIR alineados con el ECG original (demora compensada)
    zoom_fir = zoom_region[zoom_region + demora_max < cant_muestras]

    plt.plot(zoom_fir, ecg_win_hann[zoom_fir + demora_hann],
         label='FIR Hann')

    plt.plot(zoom_fir, ecg_firls[zoom_fir + demora_firls],
         label='FIR LS')

    plt.title('Región con ruido: ' + str(inicio) + ' a ' + str(fin))
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

    # plt.plot(zoom_region, ecg_butt[zoom_region],
    #          label='Butterworth')

    # plt.plot(zoom_region, ecg_cheby2[zoom_region],
    #          label='Chebyshev II')

    # plt.plot(zoom_region, ecg_cauer[zoom_region],
    #          label='Cauer')

    # # FIR alineados con el ECG original (demora compensada)
    # zoom_fir = zoom_region[zoom_region + demora_max < cant_muestras]

    # plt.plot(zoom_fir, ecg_win_boxcar[zoom_fir + demora_box],
    #          label='FIR Boxcar')

    # plt.plot(zoom_fir, ecg_win_hamming[zoom_fir + demora_hamm],
    #          label='FIR Hamming')

    # plt.plot(zoom_fir, ecg_win_hann[zoom_fir + demora_hann],
    #          label='FIR Hann')
    
    # IIR Butterworth
    plt.plot(zoom_region, ecg_butt[zoom_region],
         label='Butterworth')

# FIR alineados con el ECG original (demora compensada)
    zoom_fir = zoom_region[zoom_region + demora_max < cant_muestras]

    plt.plot(zoom_fir, ecg_win_hann[zoom_fir + demora_hann],
         label='FIR Hann')

    plt.plot(zoom_fir, ecg_firls[zoom_fir + demora_firls],
         label='FIR LS')

    plt.title('Región sin ruido: ' + str(inicio) + ' a ' + str(fin))
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.grid()
    plt.legend()
    plt.show()

#%% Respuesta al impulso

delta_k = sig.unit_impulse(1000)

resp_imp_butt = sig.sosfilt(sos_butt, delta_k)
resp_imp_cheby2 = sig.sosfilt(sos_cheby2, delta_k)
resp_imp_cauer = sig.sosfilt(sos_cauer, delta_k)

plt.figure(figsize=(10, 4))

plt.plot(resp_imp_butt, label='Butterworth')
# plt.plot(resp_imp_cheby2, label='Chebyshev II')
# plt.plot(resp_imp_cauer, label='Cauer')

# Respuesta al impulso FIR: los coeficientes b son h[n]
# plt.plot(b_win_boxcar, label='FIR Boxcar')
# plt.plot(b_win_hamming, label='FIR Hamming')
plt.plot(b_win_hann, label='FIR Hann')
plt.plot(b_firls, label='FIR LS')

plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.title('Respuesta al impulso')
plt.grid()
plt.legend()
plt.show()
