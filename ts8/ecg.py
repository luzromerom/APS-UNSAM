# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:18:57 2026

@author: manue
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
# mat_struct = sio.loadmat('./ECG_TP4.mat')

# ecg_one_lead = mat_struct['ecg_lead']
# N = len(ecg_one_lead)

# hb_1 = mat_struct['heartbeat_pattern1']
# hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

#plt.figure()
#plt.plot(ecg_one_lead)



#f,pxx=sig.welch(x, fs=1.0, window='hann_periodic', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')


# f_def,pxx_def=sig.welch(ecg_one_lead)


# plt.figure()
# plt.plot(f_def, pxx_def)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [V²/Hz]')
# plt.title('PSD del ECG - Welch')
# plt.grid()
# plt.show()




N = len(ecg_one_lead)

K_vec = range(2, 60, 10)   

plt.figure(figsize=(10,6))

for K in K_vec:
    
    L = N // K
    
    f, pxx = sig.welch(
        ecg_one_lead,
        fs=fs_ecg,
        nperseg=L,
        noverlap=0
    )
    
    plt.plot(f, pxx, label=f'K = {K}')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [V²/Hz]')
plt.title('PSD del ECG - Método de Welch')
plt.grid()
plt.legend()
plt.xlim([0, 60])
plt.show()

