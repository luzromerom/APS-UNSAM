# # -*- coding: utf-8 -*-

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal as sig


# # ============================================================
# # Lectura del ECG
# # ============================================================

# fs_ecg = 1000  # Hz

# ecg_one_lead = np.load('ecg_sin_ruido.npy')

# N = len(ecg_one_lead)


# # ============================================================
# # Valores de K
# # ============================================================

# K_ref = 2

# K_vec = [25, 30,35,40,45,50]


# # ============================================================
# # K = 2 como referencia
# # ============================================================

# L_ref = N // K_ref
# overlap_ref = L_ref // 2

# f_ref, pxx_ref = sig.welch(
#     ecg_one_lead,
#     fs=fs_ecg,
#     window='hann',
#     nperseg=L_ref,
#     noverlap=overlap_ref,
#     nfft=N,
#     scaling='density'
# )


# # ============================================================
# # Gráfico
# # ============================================================

# plt.figure(figsize=(11, 7))


# # K = 2 de fondo
# plt.plot(
#     f_ref,
#     pxx_ref,
#     '--',
#     linewidth=1.2,
#     alpha=0.4,
#     color='gray',
#     label='K = 2 (referencia)'
# )


# # ============================================================
# # Comparación de K
# # ============================================================

# for K in K_vec:

#     L = N // K
#     overlap = L // 2

#     f, pxx = sig.welch(
#         ecg_one_lead,
#         fs=fs_ecg,
#         window='hann',
#         nperseg=L,
#         noverlap=overlap,
#         nfft=N,
#         scaling='density'
#     )

#     plt.plot(
#         f,
#         pxx,
#         linewidth=1.8,
#         label=f'K = {K}'
#     )

#     # Cantidad real de segmentos utilizados
#     M = 1 + (N - L) // (L - overlap)

#     print(
#         f'K = {K:2d} | '
#         f'L = {L:4d} | '
#         f'M real = {M:2d} | '
#         f'Df = {fs_ecg/L:.3f} Hz'
#     )


# # ============================================================
# # Configuración
# # ============================================================

# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD')
# plt.title('PSD del ECG - Comparación fina de K')

# # Zoom donde están las diferencias importantes
# #plt.xlim([0, 6])

# plt.grid()
# plt.legend()

# plt.tight_layout()
# plt.show()
# # PPG
# fs_ppg = 400
# ppg = np.load('ppg_sin_ruido.npy')
# N_ppg = len(ppg)

# # comparo distintos valores de K
# K_ref = 2
# K_vec = [20, 25, 30, 35, 40, 45]

# L_ref = N_ppg // K_ref
# overlap_ref = L_ref // 2
# f_ref, pxx_ref = sig.welch(ppg, fs=fs_ppg, window='hann', nperseg=L_ref, noverlap=overlap_ref, nfft=N_ppg, scaling='density')

# fig, ax = plt.subplots(1,2,figsize=(14,5))
# ax[0].plot(f_ref, pxx_ref, '--', linewidth=1.2, alpha=0.4, color='gray', label='K = 2 (referencia)')
# ax[1].plot(f_ref, pxx_ref, '--', linewidth=1.2, alpha=0.4, color='gray', label='K = 2 (referencia)')

# pxx_max = 0

# for K in K_vec:
#     L = N_ppg // K
#     overlap = L // 2
#     f, pxx = sig.welch(ppg, fs=fs_ppg, window='hann', nperseg=L, noverlap=overlap, nfft=N_ppg, scaling='density')
#     ax[0].plot(f, pxx, linewidth=1.8, label=f'K = {K}')
#     ax[1].plot(f, pxx, linewidth=1.8, label=f'K = {K}')
#     pxx_max = max(pxx_max, np.max(pxx))
#     M = 1 + (N_ppg - L) // (L - overlap)
#     print(f'K = {K:2d} | L = {L:4d} | M real = {M:2d} | Df = {fs_ppg/L:.3f} Hz')

# ax[0].set_xlim([-0.5,50])
# ax[1].set_xlim([0,10])
# ax[0].set_ylim([-0.05*pxx_max,1.08*pxx_max])
# ax[1].set_ylim([-0.05*pxx_max,1.08*pxx_max])

# ax[0].set_xlabel('Frecuencia [Hz]')
# ax[1].set_xlabel('Frecuencia [Hz]')
# ax[0].set_ylabel('DEP')
# ax[1].set_ylabel('DEP')
# ax[0].set_title('DEP del PPG - comparación de K')
# ax[1].set_title('DEP del PPG - zoom en bajas frecuencias')
# ax[0].grid()
# ax[1].grid()
# ax[0].legend()
# ax[1].legend()

# plt.tight_layout()
# plt.show()

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio


#%% PPG

fs_ppg = 400
ppg = np.load('ppg_sin_ruido.npy')

N = len(ppg)
K_vec = [2,20,35,45,55,70]

print(f'\nPPG | N = {N} | fs = {fs_ppg} Hz | duración = {N/fs_ppg:.2f} s')

plt.figure(figsize=(10,6))

for K in K_vec:
    L = N // K
    f, pxx = sig.welch(ppg, fs=fs_ppg, window='hann', nperseg=L, noverlap=L//2, nfft=N, scaling='density')
    if K == 2:
        plt.plot(f, pxx, '--', color='gray', alpha=0.6, label='K = 2 (referencia)')
    else:
        plt.plot(f, pxx, label=f'K = {K}')
    print(f'K = {K:3d} | L = {L:5d} | Df = {fs_ppg/L:.3f} Hz')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [W/Hz]')
plt.title('PSD del PPG - Método de Welch')
plt.grid()
plt.legend()
plt.show()


#%% Audio - La cucaracha

fs_audio1, wav_data1 = sio.wavfile.read('la cucaracha.wav')

N = len(wav_data1)
K_vec = [2,30,45,60]

print(f'\nLA CUCARACHA | N = {N} | fs = {fs_audio1} Hz | duración = {N/fs_audio1:.2f} s')

plt.figure(figsize=(10,6))

for K in K_vec:
    L = N // K
    f, pxx = sig.welch(wav_data1, fs=fs_audio1, window='hann', nperseg=L, noverlap=L//2, nfft=N, scaling='density')
    if K == 2:
        plt.plot(f, pxx, '--', color='gray', alpha=0.6, label='K = 2 (referencia)')
    else:
        plt.plot(f, pxx, label=f'K = {K}')
    print(f'K = {K:3d} | L = {L:5d} | Df = {fs_audio1/L:.2f} Hz')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [W/Hz]')
plt.title('PSD del audio - La cucaracha')
plt.grid()
plt.legend()
plt.show()


#%% Audio - Prueba PSD

fs_audio2, wav_data2 = sio.wavfile.read('prueba psd.wav')

N = len(wav_data2)
K_vec = [2,30,70,85,100,200]

print(f'\nPRUEBA PSD | N = {N} | fs = {fs_audio2} Hz | duración = {N/fs_audio2:.2f} s')

plt.figure(figsize=(10,6))

for K in K_vec:
    L = N // K
    f, pxx = sig.welch(wav_data2, fs=fs_audio2, window='hann', nperseg=L, noverlap=L//2, nfft=N, scaling='density')
    if K == 2:
        plt.plot(f, pxx, '--', color='gray', alpha=0.6, label='K = 2 (referencia)')
    else:
        plt.plot(f, pxx, label=f'K = {K}')
    print(f'K = {K:3d} | L = {L:5d} | Df = {fs_audio2/L:.2f} Hz')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [W/Hz]')
plt.title('PSD del audio - Prueba PSD')
plt.grid()
plt.legend()
plt.show()


#%% Audio - Silbido

fs_audio3, wav_data3 = sio.wavfile.read('silbido.wav')

N = len(wav_data3)
K_vec = [30,60,80,150]

print(f'\nSILBIDO | N = {N} | fs = {fs_audio3} Hz | duración = {N/fs_audio3:.2f} s')

plt.figure(figsize=(10,6))

for K in K_vec:
    L = N // K
    f, pxx = sig.welch(wav_data3, fs=fs_audio3, window='hann', nperseg=L, noverlap=L//2, nfft=N, scaling='density')
    if K == 2:
        plt.plot(f, pxx, '--', color='gray', alpha=0.6, label='K = 2 (referencia)')
    else:
        plt.plot(f, pxx, label=f'K = {K}')
    print(f'K = {K:3d} | L = {L:5d} | Df = {fs_audio3/L:.2f} Hz')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [W/Hz]')
plt.title('PSD del audio - Silbido')
plt.grid()
plt.legend()
plt.show()