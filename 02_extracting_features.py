import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew

PATH = '../nasa-bearing-rul/2nd_test/'
files = sorted(os.listdir(PATH))

def extract_features(signal):

    # time domain features

    rms = np.sqrt(np.mean(signal**2))

    kurt = kurtosis(signal)

    skewness = skew(signal)

    peak_to_peak = signal.max() - signal.min()
    
    peak = np.max(np.abs(signal))

    crest_factor = peak / rms if rms != 0 else 0

    std = np.std(signal)

    shape_factor = rms / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0

    # frequency domain features

    N = len(signal)

    yf = np.abs(fft(signal))[:N//2]

    xf = fftfreq(N, 1/20000)[:N//2]  # sampling freq = 20kHz

    dominant_freq = xf[np.argmax(yf)]

    freq_energy = np.sum(yf**2)

    return {
        'rms': rms,
        'kurtosis': kurt,
        'skewness': skewness,
        'peak_to_peak': peak_to_peak,
        'peak': peak,
        'crest_factor': crest_factor,
        'standard deviation':std,
        'shape_factor': shape_factor,
        'dominant_freq': dominant_freq,
        'freq_energy': freq_energy
    }

# process all files — focus on B1 (the failing bearing)
print("Extracting features from all 984 files...")
records = []

for i, f in enumerate(files):
    df = pd.read_csv(PATH + f, sep='\t', header=None)
    df.columns = ['B1', 'B2', 'B3', 'B4']
    
    features = extract_features(df['B1'].values)
    features['file'] = f
    features['file_index'] = i
    features['RUL'] = len(files) - i  # remaining files until failure
    records.append(features)

    if i % 100 == 0:
        print(f"  Processed {i}/{len(files)} files...")

feature_df = pd.DataFrame(records)
feature_df.to_csv('features.csv', index=False)
print(f"\nDone. Feature matrix shape: {feature_df.shape}")
print(feature_df.head())