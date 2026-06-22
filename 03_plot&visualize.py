import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

feature_df = pd.read_csv(r'C:\Users\pg123\OneDrive\Desktop\nasa-bearing-rul\features.csv')
PATH = '../nasa-bearing-rul/2nd_test/'
files = sorted(os.listdir(PATH))

# --- PLOT 1: Degradation timeline ---
fig, ax = plt.subplots(3, 1, figsize=(14, 10))

ax[0].plot(feature_df['file_index'], 
             feature_df['rms'], color='steelblue')
ax[0].set_title('RMS Vibration Over Time — Bearing 1')
ax[0].set_ylabel('RMS (g)')
ax[0].axvline(x=900, color='red', 
                linestyle='--', label='Failure zone')
ax[0].legend()

ax[1].plot(feature_df['file_index'], 
             feature_df['kurtosis'], color='darkorange')
ax[1].set_title('Kurtosis Over Time')
ax[1].set_ylabel('Kurtosis')
ax[1].axvline(x=900, color='red', linestyle='--')

ax[2].plot(feature_df['file_index'], 
             feature_df['peak_to_peak'], color='green')
ax[2].set_title('Peak-to-Peak Over Time')
ax[2].set_ylabel('Peak-to-Peak (g)')
ax[2].set_xlabel('File Index (each step = 10 minutes)')
ax[2].axvline(x=900, color='red', linestyle='--')

plt.tight_layout()
plt.savefig('plots/degradation_timeline.png', dpi=150)
plt.close()
print("Plot 1 saved")

# --- PLOT 2: FFT healthy vs failure ---
df_healthy = pd.read_csv(PATH + files[0], sep='\t', header=None)
df_healthy.columns = ['B1','B2','B3','B4']

df_failure = pd.read_csv(PATH + files[960], sep='\t', header=None)
df_failure.columns = ['B1','B2','B3','B4']

fig, ax = plt.subplots(2, 1, figsize=(14, 10))

# time domain
ax[0].plot(df_healthy['B1'].values[::40], 
             label='Healthy', color='steelblue', alpha=0.8)
ax[0].plot(df_failure['B1'].values[::40], 
             label='Failure', color='red', alpha=0.8)
ax[0].set_title('Time Domain — Healthy vs Failure')
ax[0].set_ylabel('Acceleration (g)')
ax[0].legend()

# frequency domain
N = 512 # as we downsampled taking every 40th reading 
for signal, label, color in [
    (df_healthy['B1'].values, 'Healthy', 'steelblue'),
    (df_failure['B1'].values, 'Failure', 'red')
]:
    yf = np.abs(fft(signal))[:N//2]
    xf = fftfreq(N, 1/512)[:N//2]
    ax[1].plot(xf[:256], yf[:256], label=label, color=color, alpha=0.8)

ax[1].set_title('FFT Spectrum — Healthy vs Failure')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Amplitude')
ax[1].legend()

plt.tight_layout()
plt.savefig('plots/fft_comparison.png', dpi=150)
plt.close()
print("Plot 2 saved")